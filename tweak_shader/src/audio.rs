use rodio::{source::Source, Decoder, OutputStream, Sink};
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

type AudioBuffer = Arc<Mutex<Vec<f32>>>;

#[derive(Debug)]
pub struct AudioLoaderError(pub String);

impl fmt::Display for AudioLoaderError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "AudioLoaderError: {}", self.0)
    }
}

impl Error for AudioLoaderError {}

/// An audio stream manager that provides a convenient
/// interface for loading audio files into a texture.
pub struct AudioLoader {
    presentation_buffer: AudioBuffer,
    channels: u16,
    samples: usize,
    sink: Sink,
    _stream: OutputStream,
    fft: bool,
}

impl AudioLoader {
    /// Construct a new [AudioLoader] and begin playing audio out for the default audio interface,
    /// The samples per frame will be capped by `max_samples` if present. an FFT will be performed
    /// on all output, resulting in `max_samples` buckets of output if `fft` is passed in true.
    /// `initial_volume` sets the volume the audio plays at initially with 1.0 being normal and 0.0
    /// being muted.
    pub fn new<P: AsRef<Path>>(
        path: P,
        fft: bool,
        max_samples: Option<u32>,
    ) -> Result<Self, AudioLoaderError> {
        // Load a sound from the provided file
        let decoder = Decoder::new(File::open(path).map_err(|e| AudioLoaderError(e.to_string()))?)
            .map_err(|e| AudioLoaderError(e.to_string()))?;

        let first_frame = decoder.current_frame_len().unwrap_or(1024);
        let channels = decoder.channels();

        let samples = max_samples
            .map(|m| usize::min(m as usize, first_frame))
            .unwrap_or(first_frame);

        let (_stream, stream_handle) =
            OutputStream::try_default().map_err(|e| AudioLoaderError(e.to_string()))?;

        let presentation_buffer = Arc::new(Mutex::new(vec![0.0; samples]));
        let buffer_clone = presentation_buffer.clone();

        let period = Duration::from_secs_f64(1.0 / 60.0);

        let mut planner = FftPlanner::new();
        let fft_planner = planner.plan_fft_forward(samples);

        let decoder = decoder
            .repeat_infinite()
            .buffered()
            .convert_samples()
            .periodic_access(period, move |src| {
                let clone = src.clone();
                let mut buffer = buffer_clone.lock().unwrap();

                let samples_to_copy = clone.current_frame_len().unwrap_or(0);
                let samples_to_pad = buffer.len().saturating_sub(samples_to_copy);
                let channels = src.channels();
                let channel_width = samples_to_copy / channels as usize;

                // Deinterlace values padding with 0
                for (i, val) in clone
                    .take(samples_to_copy)
                    .chain(std::iter::repeat(0.0).take(samples_to_pad))
                    .enumerate()
                {
                    // align planar
                    let i = if !fft {
                        ((i % channels as usize) * channel_width) + (i / channels as usize)
                    } else {
                        i
                    };
                    buffer[i] = val;
                }

                if fft {
                    let mut input_complex: Vec<Complex<f32>> =
                        buffer.iter().map(|&x| Complex::new(x, 0.0)).collect();
                    fft_planner.process(&mut input_complex);

                    for (i, &complex) in input_complex.iter().enumerate() {
                        buffer[i] = complex.norm();
                    }
                }
            });
        // Play the sound directly on the device
        let sink = Sink::try_new(&stream_handle).unwrap();
        sink.append(decoder);

        Ok(Self {
            presentation_buffer,
            channels,
            samples: samples / channels as usize,
            fft,
            sink,
            _stream,
        })
    }

    /// Pauses the audio output
    pub fn pause(&mut self) {
        self.sink.pause();
    }

    /// unpauses the audio output
    pub fn play(&mut self) {
        self.sink.play();
    }

    /// Sets the volume of the audio ouptut, 1.0 being
    /// standard. 0.0 being completely muted and everything else being
    /// loud.
    pub fn set_volume(&mut self, volume: f32) {
        self.sink.set_volume(volume.clamp(0.0, 3.0));
    }

    /// Returns the number of audio channels in the output stream.
    pub fn channels(&self) -> u16 {
        self.channels
    }

    /// Returns true if calls to [`AudioLoader::present`]
    /// run an FFT on the data before presenting it.
    pub fn is_running_fft(&self) -> bool {
        self.fft
    }

    /// Returns the max number of samplers per frame.
    /// This number is based off of the first frame
    pub fn samples(&self) -> usize {
        self.samples
    }

    /// Returns a buffer containing a recent sample of audio data
    /// padded with zeros to extend to a uniform length.
    pub fn present(&self) -> AudioBuffer {
        self.presentation_buffer.clone()
    }
}
