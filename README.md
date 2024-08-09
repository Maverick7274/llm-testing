# Large Language Model Testing

This repository contains the code for testing the Large Language Model (LLM) developed by me. The LLM is a transformer-based model that is trained on a large corpus of text data. The model is capable of generating human-like text and can be fine-tuned for specific tasks.

## How to run the LLM

* To run the LLM, you need to have the following dependencies installed:

```plaintext
matplotlib
numpy
pylzma
ipykernel
jupyter
torch==2.0.0+cu118 -f https://download.pytorch.org/whl/cu118
```

* Before installing the dependencies, you need to create a virtual environment. You can create a virtual environment using the following command:

```bash
python -m venv venv
```

* After creating the virtual environment, you can activate it using the following command:

```bash
source venv/bin/activate
```

> **Note:** The above command is for Linux and MacOS.

* If you are using Windows, you can activate the virtual environment using the following command:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUse
.\venv\Scripts\Activate.ps1
```

* You can install these dependencies using the following command:

```bash
pip install -r requirements.txt
```

* After installing the dependencies, you will need to install the venv as a Jupyter kernel. You can do this using the following command:

```bash
python -m ipykernel install --user --name=venv --display-name "your-display-name"
```

---

### Overview of the Transformer Architecture

The Transformer architecture, introduced in the paper "Attention is All You Need" by Vaswani et al. in 2017, revolutionized the field of natural language processing (NLP). Unlike traditional sequential models like RNNs (Recurrent Neural Networks), the Transformer does not rely on sequence order and can process entire sequences in parallel, making it highly efficient and effective, especially for large-scale tasks.

Transformers are the backbone of many modern large language models (LLMs) like GPT, BERT, and T5. They are designed around a self-attention mechanism, allowing the model to weigh the importance of different words in a sentence relative to each other.

### Core Components of the Transformer

The Transformer architecture consists of two main parts:

1. **Encoder**: Processes the input sequence.
2. **Decoder**: Generates the output sequence.

Each part consists of multiple layers, typically with 6 to 12 layers each. However, LLMs can have hundreds of layers.

#### 1. **The Encoder**

Each encoder layer has two main components:

* **Self-Attention Mechanism**
* **Feed-Forward Neural Network**

**Self-Attention Mechanism:**

Self-attention allows the model to focus on different parts of the input sequence when processing a given word. The key idea is to compute a representation for each word that incorporates contextual information from the entire sequence.

For a given sequence of input embeddings $( X = [x_1, x_2, \dots, x_n] )$, the self-attention mechanism computes three vectors for each word:

* **Query $( Q )$**: A vector that represents the word we are currently processing.
* **Key $( K )$**: A vector that represents the words in the sequence we are comparing against.
* **Value $( V )$**: A vector that contains the contextual information of the word.

These vectors are computed using learned weight matrices:

$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$

### Overview of the Transformer Architecture

The Transformer architecture, introduced in the paper "Attention is All You Need" by Vaswani et al. in 2017, revolutionized the field of natural language processing (NLP). Unlike traditional sequential models like RNNs (Recurrent Neural Networks), the Transformer does not rely on sequence order and can process entire sequences in parallel, making it highly efficient and effective, especially for large-scale tasks.

Transformers are the backbone of many modern large language models (LLMs) like GPT, BERT, and T5. They are designed around a self-attention mechanism, allowing the model to weigh the importance of different words in a sentence relative to each other.

### Core Components of the Transformer

The Transformer architecture consists of two main parts:

1. **Encoder**: Processes the input sequence.
2. **Decoder**: Generates the output sequence.

Each part consists of multiple layers, typically with 6 to 12 layers each. However, LLMs can have hundreds of layers.

#### 1. **The Encoder**

Each encoder layer has two main components:

* **Self-Attention Mechanism**
* **Feed-Forward Neural Network**

**Self-Attention Mechanism:**

Self-attention allows the model to focus on different parts of the input sequence when processing a given word. The key idea is to compute a representation for each word that incorporates contextual information from the entire sequence.

For a given sequence of input embeddings $( X = [x_1, x_2, \dots, x_n] )$, the self-attention mechanism computes three vectors for each word:

* **Query $( Q )$**: A vector that represents the word we are currently processing.
* **Key $( K )$**: A vector that represents the words in the sequence we are comparing against.
* **Value $( V )$**: A vector that contains the contextual information of the word.

These vectors are computed using learned weight matrices:

$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$

where $( W_Q )$, $( W_K )$, and $( W_V )$ are the weight matrices for the queries, keys, and values, respectively.

Next, we compute the attention scores between the current word and all other words in the sequence using a dot product:

$\text{Attention Scores} = QK^T$

These scores are then scaled by the square root of the dimension of the key vectors $( d_k )$ to stabilize the gradients during training:

$\text{Scaled Attention Scores} = \frac{QK^T}{\sqrt{d_k}}$

We apply a softmax function to convert these scores into probabilities:

$\text{Attention Weights} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$

Finally, we use these weights to compute a weighted sum of the value vectors:

$\text{Attention Output} = \text{Attention Weights} \times V$

**Multi-Head Attention:**

Instead of computing a single set of attention scores, the Transformer uses multiple attention heads. Each head learns a different aspect of the sequence's relationships. The outputs from these heads are concatenated and linearly transformed:

$\text{Multi-Head Output} = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h)W_O$

where $( W_O )$ is a learned weight matrix.

**Feed-Forward Neural Network:**

After the multi-head attention, the output is passed through a feed-forward neural network. This network typically consists of two linear transformations with a ReLU activation in between:

$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$

This is applied independently to each position in the sequence.

**Residual Connections and Layer Normalization:**

Each sub-layer in the encoder is followed by a residual connection and layer normalization. The output of each sub-layer is:

$\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))$

where $( x )$ is the input to the sub-layer and $(\text{Sublayer}(x))$ is the output of the self-attention or feed-forward network.

#### 2. **The Decoder**

The decoder is similar to the encoder but includes an additional cross-attention mechanism that allows it to focus on relevant parts of the input sequence during generation.

The decoder layer consists of:

* **Masked Self-Attention**: Prevents the decoder from peeking at future tokens during training.
* **Cross-Attention**: Attends to the output of the encoder.
* **Feed-Forward Neural Network**

The cross-attention mechanism is similar to the self-attention mechanism but uses the encoder's output as the key and value vectors.

### Mathematical Formulation

Let's delve deeper into the mathematical operations within the Transformer:

#### **Scaled Dot-Product Attention**

Given an input sequence of length $( n )$ with dimension $( d_{\text{model}} )$, the attention mechanism can be summarized as follows:

1. **Linear Transformations**:

$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$

where $( X )$ is the input sequence of shape $( (n, d_{\text{model}}) )$ and $( W_Q, W_K, W_V )$ are matrices of shape $( (d_{\text{model}}, d_k) )$, with $( d_k )$ being the dimension of the queries and keys.

2. **Attention Scores**:

$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$

where $( QK^T )$ results in a matrix of shape $( (n, n) )$, representing the attention scores between each pair of words.

#### **Multi-Head Attention**

For each of the $( h )$ heads, the process is:

1. **Individual Attention Heads**:

$\text{head}_i = \text{Attention}(Q_i, K_i, V_i)$

where $( Q_i, K_i, V_i )$ are the projections for the $( i )^{th}$ head.

2. **Concatenation and Final Linear Transformation**:

$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W_O$

where $( W_O )$ is the output projection matrix.

#### **Positional Encoding**

Since the Transformer does not inherently understand the order of words, a positional encoding is added to the input embeddings. These encodings inject information about the position of each word in the sequence.

Positional encodings are computed as:

$\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right)
\quad \text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right)$

where $( pos )$ is the position and $( i )$ is the dimension index.

These encodings are added to the input embeddings $( X )$ before passing them to the encoder.

### Example Calculation

Let's consider a simple example with:

* Sequence length $( n = 3 )$
* Model dimension $( d_{\text{model}} = 4 )$
* Number of heads $( h = 2 )$
* Key/Query/Value dimension $( d_k = d_v = 2 )$

Assume the input sequence $( X )$ is:

$X = \begin{bmatrix}
1 & 0 & 1 & 0 \\
0 & 1 & 0 & 1 \\
1 & 1 & 0 & 0
\end{bmatrix}$

The weight matrices for the queries, keys, and values are:

$W_Q = W_K = W_V = \begin{bmatrix}
1 & 0 \\
0 & 1 \\
1 & 1 \\
0 & 0
\end{bmatrix}$

1. **Compute Queries, Keys, Values**:

$Q = K = V = X \times W_Q = \begin{bmatrix}
1 & 1 \\
0 & 1 \\
1 & 1
\end{bmatrix}$

2. **Compute Attention Scores**:

$\text{Scores} = Q \times K^T = \begin{bmatrix}
2 & 1 & 2 \\
1 & 1 & 1 \\
2 & 1 & 2
\end{bmatrix}$

3. **Apply Softmax**:

$\text{Softmax(Scores)} = \begin{bmatrix}
0.422 & 0.155 & 0.422 \\
0.333 & 0.333 & 0.333 \\
0.422 & 0.155 & 0.422
\end{bmatrix}$

4. **Compute Attention Output**:

$\text{Output} = \text{Softmax(Scores)} \times V = \begin{bmatrix}
1 & 1 \\
0.667 & 1 \\
1 & 1
\end{bmatrix}$

### Conclusion

The Transformer architecture's use of self-attention mechanisms, combined with feed-forward neural networks, allows it to model complex dependencies in sequences efficiently. Its ability to process entire sequences in parallel makes it particularly powerful for large-scale NLP tasks. Understanding the mathematics behind the Transformer provides insight into why it has become the foundation for state-of-the-art models in NLP.
