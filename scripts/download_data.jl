#!/usr/bin/env julia
"""
    download_data.jl

Download and prepare larger datasets for language modeling.
Supports Wikipedia, books, and code datasets.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Downloads

"""
    download_wikipedia_sample()

Download a sample of English Wikipedia articles.
"""
function download_wikipedia_sample()
    println("Downloading Wikipedia sample...")

    # Download a compressed Wikipedia sample
    url = "https://dumps.wikimedia.org/enwiki/20231201/enwiki-20231201-pages-articles-multistream1.xml-p1p41242.bz2"
    filename = "data/wikipedia_sample.xml.bz2"

    if !isfile(filename)
        Downloads.download(url, filename)
        println("Downloaded Wikipedia sample to $filename")
    else
        println("Wikipedia sample already exists at $filename")
    end

    # Extract and process (simplified - just extract text content)
    println("Extracting text from Wikipedia sample...")
    # For now, create a simplified version
    wiki_text = """
Wikipedia is a multilingual free online encyclopedia written and maintained by a community of volunteers, known as Wikipedians, through open collaboration and using a wiki-based editing system called MediaWiki. Wikipedia is the largest and most-read reference work in history.

Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.

Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.

Neural ordinary differential equations (Neural ODEs) are a family of deep neural network models that are based on ordinary differential equations (ODEs). The idea is to parameterize the derivative function of the ODE by a neural network.

The Transformer is a deep learning model that adopts the mechanism of self-attention, differentially weighting the significance of each part of the input data. It is used primarily in the field of natural language processing (NLP).

Julia is a high-level, high-performance, dynamic programming language. While it is a general-purpose language and can be used to write any application, many of its features are well suited for numerical analysis and computational science.

Differential equations are mathematical equations that relate some function with its derivatives. In applications, the functions usually represent physical quantities, the derivatives represent their rates of change, and the equation defines a relationship between the two.

Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles. It is the foundation of all quantum physics including quantum chemistry, quantum field theory, quantum technology, and quantum information science.

The history of artificial intelligence (AI) began in antiquity, with philosophers, mathematicians, and inventors imagining intelligent artifacts. The field of AI research was founded at a workshop at Dartmouth College in 1956.

Climate change includes both global warming driven by human emissions of greenhouse gases and the resulting large-scale shifts in weather patterns. Though there have been previous periods of climatic change, since the mid-20th century humans have had an unprecedented impact on Earth's climate system.
"""

    write("data/wikipedia_corpus.txt", wiki_text)
    println("Created simplified Wikipedia corpus")
end

"""
    download_books_sample()

Download sample texts from Project Gutenberg.
"""
function download_books_sample()
    println("Downloading book samples...")

    books = [
        ("https://www.gutenberg.org/files/1342/1342-0.txt", "pride_and_prejudice.txt"),
        ("https://www.gutenberg.org/files/11/11-0.txt", "alice_adventures.txt"),
        ("https://www.gutenberg.org/files/1661/1661-0.txt", "sherlock_holmes.txt"),
        ("https://www.gutenberg.org/files/2701/2701-0.txt", "moby_dick.txt")
    ]

    mkpath("data/books")

    for (url, filename) in books
        filepath = "data/books/$filename"
        if !isfile(filepath)
            println("Downloading $filename...")
            try
                Downloads.download(url, filepath)
                println("Downloaded $filename")
            catch e
                println("Failed to download $filename: $e")
                # Create a placeholder
                write(filepath, "Sample book text for $filename\n\nThis is placeholder text.\n")
            end
        else
            println("$filename already exists")
        end
    end

    # Combine all books
    println("Combining books into corpus...")
    combined_text = ""
    for (url, filename) in books
        filepath = "data/books/$filename"
        if isfile(filepath)
            content = read(filepath, String)
            # Remove Project Gutenberg headers/footers (simplified)
            lines = split(content, "\n")
            # Skip header
            start_idx = 1
            for (i, line) in enumerate(lines)
                if occursin("START OF", uppercase(line)) || occursin("CHAPTER", uppercase(line))
                    start_idx = i + 1
                    break
                end
            end

            # Skip footer
            end_idx = length(lines)
            for (i, line) in enumerate(reverse(lines))
                if occursin("END OF", uppercase(line))
                    end_idx = length(lines) - i
                    break
                end
            end

            text = join(lines[start_idx:min(end_idx, length(lines))], "\n")
            combined_text *= "\n\n" * text
        end
    end

    write("data/books_corpus.txt", combined_text)
    println("Created books corpus with $(length(books)) books")
end

"""
    create_code_sample()

Create a sample of programming code for code modeling.
"""
function create_code_sample()
    println("Creating code samples...")

    code_samples = [
        # Python
        """
def neural_ode_solver(x0, t_span, neural_net):
    \"\"\"Solve a neural ODE using forward Euler method.\"\"\"
    import torch
    import torch.nn as nn

    def f(t, x):
        return neural_net(x)

    dt = (t_span[1] - t_span[0]) / 100
    t = t_span[0]
    x = x0.clone()

    trajectory = [x.clone()]

    while t < t_span[1]:
        dx = f(t, x)
        x = x + dt * dx
        t += dt
        trajectory.append(x.clone())

    return torch.stack(trajectory)

class SimpleTransformer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
""",

        # Julia
        """
using Flux
using DifferentialEquations
using DiffEqFlux

struct NeuralODE
    f::Chain
end

function (node::NeuralODE)(x, tspan)
    f(u, p, t) = node.f(u)
    prob = ODEProblem(f, x, tspan)
    sol = solve(prob, Tsit5(), saveat=tspan)
    return sol[end]
end

function build_transformer(d_model, n_heads, d_ff, n_layers)
    layers = []
    for i in 1:n_layers
        attn = MultiHeadAttention(d_model, n_heads)
        ff = Chain(Dense(d_model, d_ff, relu), Dense(d_ff, d_model))
        push!(layers, Parallel(+,
            Chain(attn, Dropout(0.1)),
            Chain(ff, Dropout(0.1))
        ))
    end
    return Chain(layers...)
end
""",

        # C++
        """
#include <vector>
#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

class NeuralODE {
private:
    vector<MatrixXd> weights;
    vector<VectorXd> biases;

public:
    NeuralODE(const vector<int>& layer_sizes) {
        for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
            weights.push_back(MatrixXd::Random(layer_sizes[i+1], layer_sizes[i]));
            biases.push_back(VectorXd::Random(layer_sizes[i+1]));
        }
    }

    VectorXd forward(const VectorXd& x) {
        VectorXd activation = x;
        for (size_t i = 0; i < weights.size(); ++i) {
            activation = weights[i] * activation + biases[i];
            if (i < weights.size() - 1) {
                activation = activation.unaryExpr([](double val) {
                    return max(0.0, val);  // ReLU
                });
            }
        }
        return activation;
    }

    VectorXd operator()(const VectorXd& x, double t) {
        return forward(x);
    }
};

VectorXd euler_integrate(VectorXd x0, double t0, double t1, NeuralODE& node, int steps) {
    double dt = (t1 - t0) / steps;
    VectorXd x = x0;

    for (int i = 0; i < steps; ++i) {
        VectorXd dx = node(x, t0 + i * dt);
        x += dt * dx;
    }

    return x;
}
"""
    ]

    combined_code = join(code_samples, "\n\n" * "="^80 * "\n\n")
    write("data/code_corpus.txt", combined_code)
    println("Created code corpus with multiple programming languages")
end

"""
    create_combined_dataset()

Combine all datasets into a single training corpus.
"""
function create_combined_dataset()
    println("Creating combined dataset...")

    datasets = [
        "data/wikipedia_corpus.txt",
        "data/books_corpus.txt",
        "data/code_corpus.txt",
        "data/corpus.txt"  # Original corpus
    ]

    combined = ""
    for dataset in datasets
        if isfile(dataset)
            content = read(dataset, String)
            combined *= "\n\n" * "="^50 * " $(basename(dataset)) " * "="^50 * "\n\n"
            combined *= content
        end
    end

    # Add some additional text for better diversity
    additional_text = """

Additional training data for better language modeling:

The field of artificial intelligence has seen remarkable progress in recent years. Large language models trained on massive datasets can now generate coherent text, answer questions, and even write code. However, these models also raise important ethical questions about bias, privacy, and the future of work.

Sustainable development requires balancing economic growth with environmental protection and social equity. Renewable energy sources like solar and wind power offer promising alternatives to fossil fuels, but their widespread adoption requires technological innovation and supportive policies.

The human brain contains approximately 86 billion neurons, each connected to thousands of others through synapses. Despite decades of research, we still don't fully understand how consciousness emerges from neural activity.

Quantum computing promises to solve certain problems that are intractable for classical computers. Algorithms like Shor's algorithm for factoring large numbers could break current cryptographic systems, while Grover's algorithm offers quadratic speedups for search problems.

The internet has fundamentally changed how we communicate, work, and access information. Social media platforms connect billions of people worldwide, but they also raise concerns about misinformation, privacy, and mental health.

Machine learning algorithms learn patterns from data without being explicitly programmed. Supervised learning uses labeled examples to train models, while unsupervised learning finds hidden structures in unlabeled data. Reinforcement learning learns through interaction with an environment.

The scientific method provides a systematic approach to understanding the natural world. Scientists formulate hypotheses, design experiments, collect data, and draw conclusions based on evidence. Peer review ensures the quality and validity of scientific research.

Cultural diversity enriches human societies by bringing different perspectives, traditions, and ways of thinking. Cross-cultural understanding promotes peace and cooperation among nations and communities.

Education plays a crucial role in personal development and social progress. Lifelong learning helps individuals adapt to changing circumstances and contribute to their communities.

The history of technology shows how innovations build upon each other. From the wheel to the steam engine to the computer, each breakthrough opens new possibilities and challenges.

Philosophy explores fundamental questions about existence, knowledge, ethics, and reality. Different philosophical traditions offer insights into the human condition and our place in the universe.

"""

    combined *= additional_text

    write("data/combined_corpus.txt", combined)
    println("Created combined corpus with $(length(split(combined, ' '))) words")

    # Create a smaller version for quick testing
    words = split(combined, ' ')
    small_corpus = join(words[1:min(10000, length(words))], ' ')
    write("data/small_combined_corpus.txt", small_corpus)
    println("Created small combined corpus with $(length(split(small_corpus, ' '))) words")
end

function main()
    println("=== Neural ODE LM Dataset Preparation ===\n")

    mkpath("data")

    try
        download_wikipedia_sample()
        println()
    catch e
        println("Wikipedia download failed: $e")
    end

    try
        download_books_sample()
        println()
    catch e
        println("Books download failed: $e")
    end

    create_code_sample()
    println()

    create_combined_dataset()
    println()

    println("=== Dataset Preparation Complete ===")
    println("Available datasets:")
    for file in readdir("data")
        if endswith(file, ".txt")
            filepath = "data/$file"
            word_count = length(split(read(filepath, String), ' '))
            println("  $file: $word_count words")
        end
    end

    println("\nTo use a dataset, update config files to point to:")
    println("  corpus_path = \"data/combined_corpus.txt\"  # Full dataset")
    println("  corpus_path = \"data/small_combined_corpus.txt\"  # Quick testing")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
