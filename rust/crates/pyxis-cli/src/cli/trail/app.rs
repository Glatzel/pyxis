use std::collections::VecDeque;

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, MouseEvent};
use rax_nmea::data::{Identifier, Talker};
mod status;
pub use super::app::status::STATUS;
use super::tab::{ITab, Tab, TabCoord, TabInfo, TabNmea, TabSettings};
use crate::SETTINGS;
pub struct App {
    pub status: usize,

    pub raw_nmea: VecDeque<(Talker, Identifier, String)>,

    pub tab: Tab,
    pub tab_info: TabInfo,
    pub tab_coord: TabCoord,
    pub tab_nmea: TabNmea,
    pub tab_settings: TabSettings,
}

impl App {
    pub fn new() -> miette::Result<Self> {
        // let raw_nmea = VecDeque::with_capacity(SETTINGS.lock().trail.capacity);
        Ok(Self {
            status: 0,

            raw_nmea: VecDeque::with_capacity(SETTINGS.lock().trail.capacity),

            tab: Tab::Info,
            tab_info: TabInfo::default(),
            tab_coord: TabCoord::default(),
            tab_nmea: TabNmea::default(),
            tab_settings: TabSettings::default(),
        })
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> bool {
        match (self.tab, key) {
            //global key
            (
                _,
                KeyEvent {
                    code: KeyCode::Right,
                    kind: KeyEventKind::Press,
                    ..
                },
            ) => self.next_tab(),
            (
                _,
                KeyEvent {
                    code: KeyCode::Left,
                    kind: KeyEventKind::Press,
                    ..
                },
            ) => self.prev_tab(),

            (
                _,
                KeyEvent {
                    code: KeyCode::Esc,
                    kind: KeyEventKind::Press,
                    ..
                },
            ) => return true,

            //tab key
            (Tab::Info, k) => self.tab_info.handle_key(k),
            (Tab::Coord, k) => self.tab_coord.handle_key(k),
            (Tab::Nmea, k) => self.tab_nmea.handle_key(k),
            (Tab::Settings, k) => self.tab_settings.handle_key(k),
        }
        false
    }
    pub fn handle_mouse(&mut self, mouse: MouseEvent) -> bool {
        match (self.tab, mouse) {
            (Tab::Info, mouse) => self.tab_info.handle_mouse(mouse),
            (Tab::Coord, mouse) => self.tab_coord.handle_mouse(mouse),
            (Tab::Nmea, mouse) => self.tab_nmea.handle_mouse(mouse),
            (Tab::Settings, mouse) => self.tab_settings.handle_mouse(mouse),
        }
        false
    }
    fn current_tab_name(&self) -> &'static str {
        match self.tab {
            Tab::Info => "Info",
            Tab::Coord => "Coord",
            Tab::Nmea => "NMEA",
            Tab::Settings => "Settings",
        }
    }
    fn next_tab(&mut self) {
        self.tab = match self.tab {
            Tab::Info => Tab::Coord,
            Tab::Coord => Tab::Nmea,
            Tab::Nmea => Tab::Settings,
            Tab::Settings => Tab::Info,
        };
        clerk::trace!("Switch to Tab: {}", self.current_tab_name());
    }

    fn prev_tab(&mut self) {
        self.tab = match self.tab {
            Tab::Info => Tab::Settings,
            Tab::Coord => Tab::Info,
            Tab::Nmea => Tab::Coord,
            Tab::Settings => Tab::Nmea,
        };
        clerk::trace!("Switch to Tab: {}", self.current_tab_name());
    }
    pub fn draw(
        &mut self,
        f: &mut ratatui::Frame,
        area: ratatui::layout::Rect,
    ) -> miette::Result<()> {
        match self.tab {
            Tab::Info => self.tab_info.draw(f, area, &self.raw_nmea),
            Tab::Coord => self.tab_coord.draw(f, area, &self.raw_nmea),
            Tab::Nmea => self.tab_nmea.draw(f, area, &self.raw_nmea),
            Tab::Settings => self.tab_settings.draw(f, area, &self.raw_nmea),
        }
    }
    pub fn update(&mut self, talker: Talker, identifier: Identifier, sentence: String) {
        if self.raw_nmea.len() > SETTINGS.lock().trail.capacity {
            self.raw_nmea.pop_front();
        }
        self.raw_nmea.push_back((talker, identifier, sentence));
        self.status = (self.status + 1) % 4
    }
    pub fn hint(&mut self) -> String {
        const GLOBAL_HINT: [&str; 2] = ["`←/→` Tab", "`esc` Quit"];
        let tab_hint = match self.tab {
            Tab::Info => self.tab_info.hint(),
            Tab::Coord => self.tab_coord.hint(),
            Tab::Nmea => self.tab_nmea.hint(),
            Tab::Settings => self.tab_settings.hint(),
        };
        GLOBAL_HINT
            .iter()
            .chain(tab_hint.iter())
            .copied()
            .collect::<Vec<&str>>()
            .join(" | ")
    }
}
