#include <vector>
#include <memory>
#include <string>
#include <cmath> // For activation functions

// Forward declarations for various layers
class TokenEmbedding;
class PositionalEncoding;
class MultiHeadSelfAttention;
class FeedForward;
class TransformerLayer;
class GPT2Model;

///// Token Embedding Layer
class TokenEmbedding {
public:
    TokenEmbedding(int vocab_size, int embed_dim);
    std::vector<float> embed(int token_id); // Embeds a single token

private:
    int vocab_size;
    int embed_dim;
    std::vector<std::vector<float>> embeddings;
};

///// Positional Encoding Layer
class PositionalEncoding {
public:
    PositionalEncoding(int max_seq_len, int embed_dim);
    std::vector<std::vector<float>> getEncoding(int seq_len);

private:
    int max_seq_len;
    int embed_dim;
    std::vector<std::vector<float>> encoding_matrix;
};

///// Multi-Head Self-Attention Layer
class MultiHeadSelfAttention {
public:
    MultiHeadSelfAttention(int embed_dim, int num_heads);
    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& input);

private:
    int embed_dim;
    int num_heads;
    std::vector<std::vector<std::vector<float>>> weights; // For Q, K, V
};

///// Feedforward Layer
class FeedForward {
public:
    FeedForward(int embed_dim, int hidden_dim);
    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& input);

private:
    int embed_dim;
    int hidden_dim;
    std::vector<std::vector<float>> weights1;
    std::vector<std::vector<float>> weights2;
};

///// Transformer Layer (Combines Self-Attention and Feedforward Layers)
class TransformerLayer {
public:
    TransformerLayer(int embed_dim, int num_heads, int hidden_dim);
    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& input);

private:
    MultiHeadSelfAttention self_attention;
    FeedForward feed_forward;
};

///// GPT-2 Model Class
class GPT2Model {
public:
    GPT2Model(int vocab_size, int max_seq_len, int embed_dim, int num_heads, int num_layers, int hidden_dim);
    std::vector<float> forward(const std::vector<int>& token_ids);

private:
    TokenEmbedding token_embedding;
    PositionalEncoding positional_encoding;
    std::vector<TransformerLayer> transformer_layers;
    int num_layers;
};

// Constructor implementations for each class can include setting up matrices, 
// initializing weights, and other setup work specific to the GPT-2 model.

///// Implementation of the forward pass for each layer can be defined to handle
// matrix multiplications, activations, and combining attention results with residuals.

// Main function to test the model (simplified for illustration)
// int main() {
//     int vocab_size = 50257;    // GPT-2 vocabulary size
//     int max_seq_len = 1024;    // GPT-2 max sequence length
//     int embed_dim = 768;       // GPT-2 embedding dimension for small model
//     int num_heads = 12;        // Number of attention heads
//     int num_layers = 12;       // Number of transformer layers
//     int hidden_dim = 3072;     // Feedforward hidden layer dimension

//     // Initialize GPT-2 model
//     GPT2Model model(vocab_size, max_seq_len, embed_dim, num_heads, num_layers, hidden_dim);

//     // Example input
//     std::vector<int> token_ids = {12, 24, 36, 48};  // Just some token IDs

//     // Forward pass through the model
//     std::vector<float> output = model.forward(token_ids);

//     // Output can be further processed (like converting to probabilities)
//     return 0;
// }
