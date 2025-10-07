#!/usr/bin/env python3
"""
Command Line Interface for Modern Text Summarization
==================================================

Simple CLI tool for text summarization with various options.
"""

import click
import sys
from pathlib import Path
from modern_summarizer import ModernTextSummarizer, SummarizationConfig

@click.command()
@click.option('--text', '-t', help='Text to summarize')
@click.option('--file', '-f', type=click.Path(exists=True), help='File containing text to summarize')
@click.option('--method', '-m', 
              type=click.Choice(['tfidf', 'textrank', 'bert', 'sentence_bert']),
              default='tfidf', help='Summarization method')
@click.option('--sentences', '-s', type=int, default=3, help='Maximum number of sentences')
@click.option('--min-length', type=int, default=10, help='Minimum sentence length')
@click.option('--max-length', type=int, default=200, help='Maximum sentence length')
@click.option('--no-stopwords', is_flag=True, help='Keep stopwords')
@click.option('--stem', is_flag=True, help='Stem words')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--output', '-o', type=click.Path(), help='Output file for summary')
def cli(text, file, method, sentences, min_length, max_length, no_stopwords, stem, verbose, output):
    """
    Modern Text Summarization CLI Tool
    
    Summarize text using various extractive methods.
    
    Examples:
    
    \b
    # Summarize text directly
    python cli.py -t "Your text here" -m tfidf -s 3
    
    \b
    # Summarize from file
    python cli.py -f input.txt -m textrank -s 5
    
    \b
    # Save output to file
    python cli.py -t "Your text" -o summary.txt
    """
    
    # Get input text
    if file:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                input_text = f.read()
        except Exception as e:
            click.echo(f"Error reading file: {e}", err=True)
            sys.exit(1)
    elif text:
        input_text = text
    else:
        click.echo("Please provide either --text or --file", err=True)
        sys.exit(1)
    
    if not input_text.strip():
        click.echo("No text to summarize", err=True)
        sys.exit(1)
    
    # Create configuration
    config = SummarizationConfig(
        max_sentences=sentences,
        min_sentence_length=min_length,
        max_sentence_length=max_length,
        method=method,
        remove_stopwords=not no_stopwords,
        stem_words=stem
    )
    
    if verbose:
        click.echo(f"Configuration: {config}")
        click.echo(f"Input text length: {len(input_text)} characters")
        click.echo(f"Method: {method}")
        click.echo("=" * 50)
    
    # Create summarizer and generate summary
    try:
        summarizer = ModernTextSummarizer(config)
        summary, scores = summarizer.summarize(input_text, method)
        
        # Format output
        output_text = f"Summary ({method.upper()}):\n\n"
        for i, sentence in enumerate(summary, 1):
            output_text += f"{i}. {sentence}\n"
        
        output_text += f"\nEvaluation Metrics:\n"
        output_text += f"ROUGE-1 F1: {scores['rouge1_fmeasure']:.3f}\n"
        output_text += f"ROUGE-2 F1: {scores['rouge2_fmeasure']:.3f}\n"
        output_text += f"ROUGE-L F1: {scores['rougeL_fmeasure']:.3f}\n"
        
        # Output results
        if output:
            try:
                with open(output, 'w', encoding='utf-8') as f:
                    f.write(output_text)
                click.echo(f"Summary saved to {output}")
            except Exception as e:
                click.echo(f"Error saving output: {e}", err=True)
                sys.exit(1)
        else:
            click.echo(output_text)
            
    except Exception as e:
        click.echo(f"Error during summarization: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@click.group()
def main():
    """Modern Text Summarization System"""
    pass

@main.command()
@click.option('--method', '-m', 
              type=click.Choice(['tfidf', 'textrank', 'bert', 'sentence_bert']),
              default='tfidf', help='Summarization method')
@click.option('--sentences', '-s', type=int, default=3, help='Maximum number of sentences')
def demo(method, sentences):
    """Run a demo with sample text"""
    
    sample_text = """
    Artificial Intelligence (AI) is transforming industries worldwide. 
    It is being used in healthcare to detect diseases early, in finance to prevent fraud, 
    and in manufacturing to optimize supply chains. AI-powered assistants are now part of daily life. 
    Despite its promise, AI raises ethical concerns like bias and job displacement. 
    As AI evolves, regulations and responsible development become crucial.
    """
    
    click.echo("🧠 Modern Text Summarization Demo\n")
    click.echo(f"Original text ({len(sample_text.split('.'))} sentences):")
    click.echo(sample_text.strip())
    click.echo("\n" + "="*60 + "\n")
    
    config = SummarizationConfig(max_sentences=sentences, method=method)
    summarizer = ModernTextSummarizer(config)
    
    try:
        summary, scores = summarizer.summarize(sample_text, method)
        
        click.echo(f"📝 {method.upper()} Summary:")
        for i, sentence in enumerate(summary, 1):
            click.echo(f"{i}. {sentence}")
        
        click.echo(f"\n📊 Evaluation Metrics:")
        click.echo(f"ROUGE-1 F1: {scores['rouge1_fmeasure']:.3f}")
        click.echo(f"ROUGE-2 F1: {scores['rouge2_fmeasure']:.3f}")
        click.echo(f"ROUGE-L F1: {scores['rougeL_fmeasure']:.3f}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)

@main.command()
def methods():
    """Show available summarization methods"""
    
    methods_info = {
        'tfidf': {
            'name': 'TF-IDF + Cosine Similarity',
            'description': 'Traditional approach using term frequency-inverse document frequency',
            'speed': 'Fast',
            'quality': 'Good',
            'best_for': 'General text, fast processing'
        },
        'textrank': {
            'name': 'TextRank',
            'description': 'Graph-based algorithm inspired by PageRank',
            'speed': 'Medium',
            'quality': 'Very Good',
            'best_for': 'Longer documents, graph-based analysis'
        },
        'bert': {
            'name': 'BERT',
            'description': 'Bidirectional Encoder Representations from Transformers',
            'speed': 'Slow',
            'quality': 'Excellent',
            'best_for': 'High-quality summaries, semantic understanding'
        },
        'sentence_bert': {
            'name': 'Sentence-BERT',
            'description': 'Optimized sentence embeddings for semantic similarity',
            'speed': 'Medium',
            'quality': 'Very Good',
            'best_for': 'Balanced speed and quality'
        }
    }
    
    click.echo("🧠 Available Summarization Methods\n")
    
    for method, info in methods_info.items():
        click.echo(f"📝 {info['name']} ({method})")
        click.echo(f"   Description: {info['description']}")
        click.echo(f"   Speed: {info['speed']}")
        click.echo(f"   Quality: {info['quality']}")
        click.echo(f"   Best for: {info['best_for']}")
        click.echo()

if __name__ == '__main__':
    cli()
