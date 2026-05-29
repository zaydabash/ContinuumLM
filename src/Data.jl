"""
    Data.jl

Data loading, tokenization, and batching utilities.
"""
module Data

using Random: shuffle!

export SimpleTokenizer, build_tokenizer, load_corpus, encode, decode, encode_corpus, make_batches, save_tokenizer, load_tokenizer, split_train_val

"""
    SimpleTokenizer

A simple word-level tokenizer for language modeling.
"""
struct SimpleTokenizer
    vocab::Dict{String, Int}
    idx_to_token::Vector{String}
    unk_token::Int
end

"""
    build_tokenizer(corpus::String; vocab_size::Int)

Build a simple word-level tokenizer from the given corpus.
"""
function build_tokenizer(corpus::String; vocab_size::Int = 8000)
    words = split(corpus, r"\s+|(?=[.,!?;])|(?<=[.,!?;])")
    word_counts = Dict{String, Int}()
    for word in words
        if !isempty(word)
            word_counts[word] = get(word_counts, word, 0) + 1
        end
    end

    sorted_vocab = sort(collect(word_counts), by=x->x[2], rev=true)
    
    vocab = Dict{String, Int}()
    idx_to_token = Vector{String}()
    
    push!(idx_to_token, "<UNK>")
    push!(idx_to_token, "<EOS>")
    push!(idx_to_token, "<BOS>")

    vocab["<UNK>"] = 1
    vocab["<EOS>"] = 2
    vocab["<BOS>"] = 3

    current_idx = 4
    for (word, _) in sorted_vocab
        if current_idx <= vocab_size
            vocab[word] = current_idx
            push!(idx_to_token, word)
            current_idx += 1
        else
            break
        end
    end
    
    return SimpleTokenizer(vocab, idx_to_token, vocab["<UNK>"])
end

"""
    encode(tok::SimpleTokenizer, text::String) -> Vector{Int}

Encode text to token IDs.
"""
function encode(tok::SimpleTokenizer, text::String)
    words = split(text, r"\s+|(?=[.,!?;])|(?<=[.,!?;])", keepempty=false)
    ids = [get(tok.vocab, word, tok.unk_token) for word in words]
    return ids
end

"""
    decode(tok::SimpleTokenizer, ids::Vector{Int}) -> String

Decode token IDs back to text.
"""
function decode(tok::SimpleTokenizer, ids::Vector{Int})
    words = [tok.idx_to_token[id] for id in ids]
    return join(words, " ")
end

"""
    save_tokenizer(tok, path::String)

Save a tokenizer to disk.
"""
function save_tokenizer(tok::SimpleTokenizer, path::String)
    mkpath(dirname(path))
    # One token per line; the 1-based line number is the token ID.
    # Tokens never contain whitespace (the corpus is split on whitespace and
    # punctuation), so each line holds exactly one token even when that token
    # is itself a comma. This preserves index order exactly on round-trip.
    open(path, "w") do io
        for word in tok.idx_to_token
            write(io, word, "\n")
        end
    end
end

"""
    load_tokenizer(path::String)

Load a tokenizer from disk.
"""
function load_tokenizer(path::String)
    idx_to_token = String[]
    open(path, "r") do io
        for line in eachline(io)
            # eachline strips the trailing newline; the line is the token itself.
            push!(idx_to_token, line)
        end
    end
    vocab = Dict{String, Int}()
    for (id, word) in enumerate(idx_to_token)
        vocab[word] = id
    end
    unk_token_id = get(vocab, "<UNK>", 1)
    return SimpleTokenizer(vocab, idx_to_token, unk_token_id)
end

"""
    load_corpus(path::String) -> String

Load a text corpus from disk.
"""
function load_corpus(path::String)
    if !isfile(path)
        error("Corpus file not found: $path")
    end
    open(path, "r") do io
        return read(io, String)
    end
end

"""
    encode_corpus(tok, corpus; seq_len)

Encode corpus to token ids and chunk into sequences of length `seq_len`.
Return a vector of integer arrays.
"""
function encode_corpus(tok::SimpleTokenizer, corpus::String; seq_len::Int)
    ids = encode(tok, corpus)
    n = length(ids) ÷ seq_len
    if n == 0
        error("Corpus too short for sequence length $seq_len")
    end
    ids = ids[1:(n*seq_len)]
    x = reshape(ids, (seq_len, n))
    return [Vector{Int}(col) for col in eachcol(x)]
end

"""
    make_batches(sequences, batch_size)

Take a vector of token sequences, shuffle, and group into batches.
Returns vector of (x, y) tuples for language modeling.
"""
function make_batches(seqs::Vector{Vector{Int}}, batch_size::Int)
    shuffled = copy(seqs)
    shuffle!(shuffled)
    nbatch = length(shuffled) ÷ batch_size
    if nbatch == 0
        error("Not enough sequences for batch size $batch_size")
    end

    batches = []
    for i in 1:nbatch
        batch = shuffled[(i-1)*batch_size+1 : i*batch_size]
        seq_len = length(batch[1])
        x = Array{Int}(undef, seq_len-1, batch_size)
        y = Array{Int}(undef, seq_len-1, batch_size)
        for (j, s) in enumerate(batch)
            x[:, j] = s[1:end-1]
            y[:, j] = s[2:end]
        end
        push!(batches, (x, y))
    end
    return batches
end

"""
    split_train_val(sequences, train_split)

Split sequences into train and validation sets.
"""
function split_train_val(seqs::Vector{Vector{Int}}, train_split::Float64)
    shuffle!(seqs)
    n_train = Int(floor(length(seqs) * train_split))
    return seqs[1:n_train], seqs[n_train+1:end]
end

end # module
