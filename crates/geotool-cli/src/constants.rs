use std::sync::LazyLock;

use console::Style;
// console style
pub static TRACE_STYLE: LazyLock<Style> = LazyLock::new(|| Style::new().color256(99));
pub static DEBUG_STYLE: LazyLock<Style> = LazyLock::new(|| Style::new().blue());
pub static INFO_STYLE: LazyLock<Style> = LazyLock::new(|| Style::new().green());
pub static WARN_STYLE: LazyLock<Style> = LazyLock::new(|| Style::new().yellow().bold());
pub static ERROR_STYLE: LazyLock<Style> = LazyLock::new(|| Style::new().red().bold());
