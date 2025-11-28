"""
    Data.jl

Data loading, tokenization, and batching utilities.
"""
module Data

# Random and Serialization are used via Base functions

export build_tokenizer, load_corpus, encode_corpus, make_batches, save_tokenizer, load_tokenizer, split_train_val

"""
    SimpleTokenizer

A simple word-level tokenizer for language modeling.
"""
struct SimpleTokenizer
    word_to_id::Dict{String, Int}
    id_to_word::Dict{Int, String}
    vocab_size::Int
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
    build_tokenizer(corpus::String; vocab_size::Int)

Build a simple word-level tokenizer from the given corpus.
"""
function build_tokenizer(corpus::String; vocab_size::Int = 8000)
    words = split(corpus)
    unique_words = unique(words)
    
    # Limit vocabulary size
    vocab_list = unique_words[1:min(vocab_size-2, length(unique_words))]
    
    # Create mappings (reserve 0 for padding, 1 for UNK)
    word_to_id = Dict{String, Int}()
    id_to_word = Dict{Int, String}()
    
    word_to_id["<UNK>"] = 1
    id_to_word[1] = "<UNK>"
    
    for (i, word) in enumerate(vocab_list)
        id = i + 1
        word_to_id[word] = id
        id_to_word[id] = word
    end
    
    return SimpleTokenizer(word_to_id, id_to_word, length(word_to_id))
end

"""
    encode(tok::SimpleTokenizer, text::String) -> Vector{Int}

Encode text to token IDs.
"""
function encode(tok::SimpleTokenizer, text::String)
    words = split(text)
    ids = Int[]
    for word in words
        id = get(tok.word_to_id, word, 1)  # 1 is UNK
        push!(ids, id)
    end
    return ids
end

"""
    decode(tok::SimpleTokenizer, ids::Vector{Int}) -> String

Decode token IDs back to text.
"""
function decode(tok::SimpleTokenizer, ids::Vector{Int})
    words = String[]
    for id in ids
        word = get(tok.id_to_word, id, "<UNK>")
        push!(words, word)
    end
    return join(words, " ")
end

"""
    save_tokenizer(tok, path::String)

Save a tokenizer to disk.
"""
function save_tokenizer(tok::SimpleTokenizer, path::String)
    mkpath(dirname(path))
    open(path, "w") do io
        Base.Serialization.serialize(io, tok)
    end
end

"""
    load_tokenizer(path::String)

Load a tokenizer from disk.
"""
function load_tokenizer(path::String)
    open(path, "r") do io
        return Base.Serialization.deserialize(io)
    end
end

"""
    encode_corpus(tok, corpus; seq_len)

Encode corpus to token ids and chunk into sequences of length `seq_len`.
Return a vector of integer arrays.
"""
function encode_corpus(tok::SimpleTokenizer, corpus::String; seq_len::Int)
    ids = encode(tok, corpus)
    # simple chunking: discard tail
    n = length(ids) ÷ seq_len
    if n == 0
        error("Corpus too short for sequence length $seq_len")
    end
    ids = ids[1:(n*seq_len)]
    x = reshape(ids, (seq_len, n)) # seq_len × n
    return collect(eachcol(x))     # Vector{Vector{Int}}
end

"""
    make_batches(sequences, batch_size)

Take a vector of token sequences (each Vector{Int}), group into batches,
and return an iterator over (x, y) pairs for LM training.

x: input tokens (seq_len, batch)
y: target tokens (seq_len, batch) shifted by one.
"""
function make_batches(seqs::Vector{Vector{Int}}, batch_size::Int)
    # shuffle sequences
    shuffled = copy(seqs)
    Base.Random.shuffle!(shuffled)
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
    Base.Random.shuffle!(seqs)
    n_train = Int(floor(length(seqs) * train_split))
    return seqs[1:n_train], seqs[n_train+1:end]
end

end # module
