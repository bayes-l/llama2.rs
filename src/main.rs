mod transformer;
mod tokenizer;

extern crate clap;
extern crate anyhow;

use clap::Parser;

#[derive(Parser)]
#[command(name = "llama2")]
#[command(about = "Run inference with LLaMA 2 transformer model", long_about = None)]
struct Cli {
    checkpoint: String,
    #[arg(short = 'z', long, default_value = "tokenizer.bin")]
    tokenizer: String,
    #[arg(short = 't', long, default_value = "1.0")]
    temperature: f32,
    #[arg(short = 'n', long, default_value = "256")]
    steps: usize,
    #[arg(short = 'i', long)]
    prompt: Option<String>,
}

fn main() {
    let args = Cli::parse();

    if args.temperature < 0.0 {
        eprintln!("Temperature must be non-negative");
        std::process::exit(1);
    }

}
