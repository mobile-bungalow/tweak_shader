use ffmpeg::format::Pixel;
use ffmpeg::software::scaling::{context::Context as SwScaler, flag::Flags};
use ffmpeg::util::frame::video::Video;
use ffmpeg_next as ffmpeg;

use std::fmt;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Condvar, Mutex};

type TextureBuffer = Arc<Mutex<Vec<u8>>>;

#[derive(Debug)]
pub struct VideoLoaderErr(pub String);

impl fmt::Display for VideoLoaderErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "VideoLoaderErr: {}", self.0)
    }
}

// A greedy video loader that has two buffers,
// One buffer is the "presentation buffer"
// the other buffer is the "preperation buffer"
pub struct VideoLoader {
    presentation_buffer: TextureBuffer,
    back_buffer: TextureBuffer,
    dropped: Arc<AtomicBool>,
    next_frame_ready: Arc<(Condvar, Mutex<bool>)>,
    frame_start_time: std::time::Instant,
    frame_rate: std::time::Duration,
    width: u32,
    height: u32,
}

pub struct FfmpegContext {
    pub input: ffmpeg::format::context::Input,
    pub decoder: ffmpeg::codec::decoder::Video,
    pub index: usize,
    pub scaler: SwScaler,
}

impl VideoLoader {
    /// Opens the video file at `path` and immediately starts decoding it
    pub fn init<P: AsRef<std::path::Path>>(path: P) -> Result<Self, VideoLoaderErr> {
        let format_context = ffmpeg::format::input(&path)
            .map_err(|_| VideoLoaderErr("Failed to open video file".to_owned()))?;

        let video_stream = format_context
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or_else(|| VideoLoaderErr("No video stream found".to_owned()))?;

        let index = video_stream.index();

        let codec_parameters = video_stream.parameters();
        let context_decoder = ffmpeg::codec::context::Context::from_parameters(codec_parameters)
            .map_err(|_| VideoLoaderErr("Could not create decoder".to_owned()))?;

        let decoder = context_decoder.decoder().video().unwrap();
        let width = decoder.width();
        let height = decoder.height();

        let scaler = SwScaler::get(
            decoder.format(),
            width,
            height,
            Pixel::RGBA,
            width,
            height,
            Flags::BITEXACT,
        )
        .map_err(|_| VideoLoaderErr("Could not get Software Scaler".to_owned()))?;

        let size = width as usize * height as usize * 4;
        let presentation_buffer = Arc::new(Mutex::new(vec![0; size]));
        let back_buffer = Arc::new(Mutex::new(vec![0; size]));

        let mut ffmpeg_ctx = FfmpegContext {
            input: format_context,
            decoder,
            index,
            scaler,
        };

        load_next_video_frame(&presentation_buffer, &mut ffmpeg_ctx);
        load_next_video_frame(&back_buffer, &mut ffmpeg_ctx);

        // we need to pack and unpack this because the scaler
        // is not thread safe.
        let FfmpegContext {
            input: format_context,
            decoder,
            index,
            ..
        } = ffmpeg_ctx;

        let frame_rate_rational = decoder.frame_rate().unwrap_or(ffmpeg::Rational(1, 24));
        let frame_rate_float = frame_rate_rational.0 as f64 / frame_rate_rational.1 as f64;

        // ffmpeg misreports frame rate ALOT
        // so we jsut make a best guess here defaulting to
        // the most common option.
        let frame_rate = if frame_rate_float < (1.0 / 20.0) && frame_rate_float > (1.0 / 60.0) {
            std::time::Duration::from_secs_f64(frame_rate_float)
        } else {
            std::time::Duration::from_secs_f64(1.0 / 24.0)
        };

        let dropped = Arc::new(AtomicBool::new(false));
        let dropped_clone = dropped.clone();

        let next_frame_ready = Arc::new((Condvar::new(), Mutex::new(false)));
        let next_frame_ready_clone = next_frame_ready.clone();

        let back_buffer_clone = back_buffer.clone();

        std::thread::spawn(move || {
            let scaler = SwScaler::get(
                decoder.format(),
                width,
                height,
                Pixel::RGBA,
                width,
                height,
                Flags::BITEXACT,
            )
            .unwrap();

            let mut ffmpeg_ctx = FfmpegContext {
                input: format_context,
                decoder,
                index,
                scaler,
            };

            loop {
                load_next_video_frame(&back_buffer_clone, &mut ffmpeg_ctx);

                let (ref condvar, ref lock) = &*next_frame_ready_clone;
                let mut next_frame_ready_lock = lock.lock().unwrap();
                *next_frame_ready_lock = true;

                while *next_frame_ready_lock {
                    next_frame_ready_lock = condvar.wait(next_frame_ready_lock).unwrap();

                    if dropped_clone.load(std::sync::atomic::Ordering::SeqCst) {
                        break;
                    }
                }

                if dropped_clone.load(std::sync::atomic::Ordering::SeqCst) {
                    break;
                }
            }
        });

        Ok(VideoLoader {
            frame_start_time: std::time::Instant::now(),
            frame_rate,
            presentation_buffer,
            back_buffer,
            next_frame_ready,
            dropped,
            height,
            width,
        })
    }

    fn next_frame_due(&mut self) -> bool {
        if self.frame_start_time.elapsed() > self.frame_rate {
            self.frame_start_time = std::time::Instant::now();
            true
        } else {
            false
        }
    }

    /// returns the next frame, or none if it is either not ready or the next
    /// frame is not due
    pub fn present(&mut self) -> Option<TextureBuffer> {
        let frame_due = self.next_frame_due();
        // Wait for the frame to be available.
        let (ref condvar, ref lock) = &*self.next_frame_ready;
        let mut next_frame_ready_lock = lock.lock().ok()?;

        // swap back buffer
        if *next_frame_ready_lock && frame_due {
            *next_frame_ready_lock = false;
            condvar.notify_one();
            {
                let mut back = self.back_buffer.lock().ok()?;
                let mut present = self.presentation_buffer.lock().ok()?;
                std::mem::swap(&mut *back, &mut *present);
            }

            Some(self.presentation_buffer.clone())
        } else {
            None
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }
}

impl Drop for VideoLoader {
    fn drop(&mut self) {
        self.dropped
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let (ref condvar, _) = &*self.next_frame_ready;
        condvar.notify_one();
    }
}

fn load_next_video_frame(target: &TextureBuffer, ctx: &mut FfmpegContext) {
    let FfmpegContext {
        input,
        decoder,
        index,
        scaler,
    } = ctx;

    let mut packets = input.packets().peekable();

    if packets.peek().is_none() {
        let _ = input.seek(0, 0..0);
        packets = input.packets().peekable();
    }

    for (stream, packet) in packets {
        if stream.index() == *index {
            let Ok(_) = decoder.send_packet(&packet) else {
                return;
            };

            let mut decoded = Video::empty();
            if decoder.receive_frame(&mut decoded).is_ok() {
                let mut rgb_frame = Video::empty();
                scaler.run(&decoded, &mut rgb_frame).unwrap();

                let data = rgb_frame.data(0);

                let mut buffer_lock = target.lock().unwrap();

                buffer_lock.copy_from_slice(data);

                break;
            }
        }
    }
}
