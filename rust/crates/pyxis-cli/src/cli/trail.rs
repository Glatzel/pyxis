use std::io::stdout;
use std::time::Duration;

use crossterm::event::Event;
use crossterm::execute;
use crossterm::terminal::{enable_raw_mode, *};

use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use tokio::sync::mpsc;
use tokio::task;
mod app;

mod serial;
pub mod settings;
mod tab;
mod ui;

/// Entry point of the async TUI application
pub async fn execute() -> mischief::Result<()> {
    // Check if serial port is available.
    serial::check_port()?;

    // Enable raw mode and enter alternate screen for TUI
    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen)?;

    // Set up terminal with Crossterm backend
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout))?;

    // Create async channel for receiving serial data
    let (tx, mut rx) = mpsc::channel(100);

    // Initialize the application state
    let mut app = app::App::new()?;

    // Spawn async task to read from serial port
    tokio::spawn(serial::start_serial_reader(tx));

    // Main TUI loop
    loop {
        // Redraw the UI
        terminal
            .draw(|f| match ui::draw(f, &mut app) {
                Ok(_) => (),
                Err(e) => clerk::error!("{e}"),
            })
            ?;

        // Handle input and serial updates concurrently
        tokio::select! {
            // Poll for keyboard/mouse events
            maybe_evt = poll_event(Duration::from_millis(10)) => {
                if let Ok(Some(evt)) = maybe_evt {
                    match evt {
                        Event::Key(key) => {
                            // Handle keyboard input; break loop on exit signal
                            if app.handle_key(key) {
                                break;
                            }
                        }
                        Event::Mouse(mouse_evt) => {
                            // Handle mouse input
                            app.handle_mouse(mouse_evt);
                        }
                        _ => {}
                    }
                }
            }

            // Handle incoming serial data
            Some((talker, identifier, sentence)) = rx.recv() => {
                app.update(talker, identifier, sentence);
            }
        }
    }

    // Restore terminal state before exiting
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}
async fn poll_event(timeout: Duration) -> std::io::Result<Option<Event>> {
    task::spawn_blocking(move || {
        if crossterm::event::poll(timeout)? {
            return crossterm::event::read().map(Some);
        }
        Ok(None)
    })
    .await
    .expect("join error")
}
