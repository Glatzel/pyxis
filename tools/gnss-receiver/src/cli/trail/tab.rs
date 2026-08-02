use std::collections::VecDeque;

pub use coord::TabCoord;
use crossterm::event::{KeyEvent, MouseEvent};
pub use info::TabInfo;
pub use nmea::TabNmea;
use ratatui::Frame;
use rax_nmea::data::{Identifier, Talker};
pub use settings::TabSettings;

mod coord;
mod info;
mod nmea;
mod settings;
pub trait ITab: Default {
    fn handle_key(&mut self, key: KeyEvent);
    fn handle_mouse(&mut self, mouse: MouseEvent);
    fn draw(
        &mut self,
        f: &mut Frame,
        area: ratatui::layout::Rect,
        raw_nmea: &VecDeque<(Talker, Identifier, String)>,
    ) -> mischief::Result<()>;
    fn hint(&mut self) -> &'static [&'static str];
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Tab {
    Info,
    Coord,
    Nmea,
    Settings,
}
impl Tab {
    pub const fn index(&self) -> usize {
        match self {
            Self::Info => 0,
            Self::Coord => 1,
            Self::Nmea => 2,
            Self::Settings => 3,
        }
    }
}
