use std::sync::{Arc, Mutex};

pub struct VideoLoader;

impl super::VideoLoaderTrait for VideoLoader {
    fn init<P: AsRef<std::path::Path>>(_path: P) -> Result<Self, String> {
        Err("Compiled without the video feature!".to_owned())
    }
    fn present(&self) -> Option<Arc<Mutex<Vec<u8>>>> {
        None
    }
    fn width(&self) -> u32 {
        1
    }
    fn height(&self) -> u32 {
        1
    }
}
