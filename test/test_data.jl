"""
    test_data.jl

Tests for data loading, tokenization, and batching.
"""

using Test
using NeuralODELM
using NeuralODELM.Data

function test_data()
    # Create a small test corpus
    test_corpus = """
    The quick brown fox jumps over the lazy dog.
    Machine learning is a subset of artificial intelligence.
    Neural networks are inspired by biological neurons.
    Deep learning has revolutionized many fields.
    """ |> strip
    
    # Test tokenizer building
    tok = build_tokenizer(test_corpus; vocab_size=100)
    @test tok isa Data.SimpleTokenizer
    
    # Test encoding/decoding round-trip
    ids = encode(tok, test_corpus)
    decoded = decode(tok, ids)
    @test length(ids) > 0
    @test decoded isa String
    
    # Test corpus encoding
    # seq_len=8 (rather than a larger value) so this short fixture corpus
    # yields multiple sequences -- enough for batching and an 80/20 split
    # below to both be non-empty.
    seqs = encode_corpus(tok, test_corpus; seq_len=8)
    @test length(seqs) > 0
    @test all(s -> length(s) == 8, seqs)

    # Test batching
    batches = make_batches(seqs, 2)
    batch_count = 0
    for (x, y) in batches
        batch_count += 1
        @test size(x) == (7, 2)  # seq_len-1, batch_size
        @test size(y) == (7, 2)
        @test all(x .>= 0)
        @test all(y .>= 0)
    end
    @test batch_count > 0
    
    # Test train/val split
    train_seqs, val_seqs = split_train_val(seqs, 0.8)
    @test length(train_seqs) + length(val_seqs) == length(seqs)
    @test length(train_seqs) > 0
    @test length(val_seqs) > 0
end

