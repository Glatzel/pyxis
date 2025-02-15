mod cli;

fn main() {
    geotool_algorithm::gcj02_to_wgs84_exact(121.09626935575027, 30.608604331756705, 1e-6, 30);
    cli::main();
}
