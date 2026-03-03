use super::record::Record;

pub fn output_simple(record: &Record) {
    println!(
        "{}: {}, {}: {}, {}: {}",
        record.output_x_name,
        record.output_x,
        record.output_y_name,
        record.output_y,
        record.output_z_name,
        record.output_z,
    )
}
pub fn output_plain(name: &str, records: &[Record]) {
    if name.is_empty() {
        println!("=================");
        println!("Transform Records");
        println!("=================");
    } else {
        println!("{}", "=".repeat(name.len()));
        println!("{name}");
        println!("{}", "=".repeat(name.len()));
    }

    if let Some(input) = records.first() {
        println!(
            "{}: {}, {}: {}, {}: {}",
            input.output_x_name,
            input.output_x,
            input.output_y_name,
            input.output_y,
            input.output_z_name,
            input.output_z,
        )
    }
    for record in records.iter().skip(1) {
        println!(
            r#"|-- step: {}
|-- method: {}
|-- parameter:
{}
{}
{}: {}, {}: {}, {}: {}"#,
            record.idx,
            record.method,
            serde_json::to_string_pretty(&record.parameter)
                .unwrap()
                .lines()
                .map(|line| format!("|       {line}"))
                .collect::<Vec<String>>()
                .join("\n"),
            "\u{25BC}",
            record.output_x_name,
            record.output_x,
            record.output_y_name,
            record.output_y,
            record.output_z_name,
            record.output_z,
        )
    }
}
pub fn output_json(name: &str, records: &[Record]) {
    let json_txt =
        serde_json::to_string_pretty(&serde_json::json!({"name":name, "record":records })).unwrap();
    println!("{json_txt}")
}
