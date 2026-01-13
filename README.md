# llama2.rs

A Rust implementation of [llama2.c](https://github.com/karpathy/llama2.c) for running inference on LLaMA 2 models.

## Building

To build the project, you need to have Rust installed. If you don't have Rust installed, you can get it from [rustup.rs](https://rustup.rs/).

Build the project with:

```bash
cargo build --release
```

The binary will be located at `target/release/llama2`.

## Usage

```bash
./target/release/llama2 <checkpoint_file> [OPTIONS]
```

### Options

- `-z, --tokenizer <TOKENIZER>`: Path to the tokenizer file (default: `tokenizer.bin`)
- `-t, --temperature <TEMPERATURE>`: Temperature for sampling (0.0 = greedy argmax sampling, 1.0 = sampling from distribution) (default: 1.0)
- `-n, --steps <STEPS>`: Number of steps to run for (default: 256)
- `-i, --prompt <PROMPT>`: Optional prompt to condition on

### Example

```bash
# Run with a checkpoint file
./target/release/llama2 model.bin

# Run with a custom tokenizer and prompt
./target/release/llama2 model.bin -z tokenizer.bin -i "Once upon a time" -n 100 -t 0.8
```

## Model Files

To use this implementation, you need:
1. A model checkpoint file (`.bin` format)
2. A tokenizer file (`tokenizer.bin`)

These files can be obtained by training or downloading pre-trained models compatible with the llama2.c format.

## License

MIT