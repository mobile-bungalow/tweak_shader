#[cfg(not(feature = "video"))]
mod dummy_video;

#[cfg(feature = "video")]
mod video;

#[cfg(not(feature = "video"))]
pub use dummy_video::VideoLoader;

#[cfg(feature = "video")]
pub use video::VideoLoader;

use std::sync::{Arc, Mutex};

pub trait VideoLoaderTrait: Sized {
    fn init<P: AsRef<std::path::Path>>(_path: P) -> Result<Self, String>;
    fn present(&self) -> Option<Arc<Mutex<Vec<u8>>>>;
    fn width(&self) -> u32;
    fn height(&self) -> u32;
}
