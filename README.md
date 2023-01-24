# Transformer
A transformer is a deep learning model that adopts the mechanism of self-attention, differentially weighting the significance of each part of the input data. It is used primarily in the fields of natural language processing (NLP) and computer vision (CV)


https://jalammar.github.io/illustrated-transformer/

![image](https://user-images.githubusercontent.com/122109969/214242998-323b9f47-74cd-490a-85ea-306d1ad242cc.png)

Popping open that Optimus Prime goodness, we see an encoding component, a decoding component, and connections between them.

![image](https://user-images.githubusercontent.com/122109969/214243203-3aa71eec-0e43-4a3a-be4e-6af97516736c.png)

The encoding component is a stack of encoders (the paper stacks six of them on top of each other – there’s nothing magical about the number six, one can definitely experiment with other arrangements). The decoding component is a stack of decoders of the same number.

![image](https://user-images.githubusercontent.com/122109969/214243365-b0942cd3-32e8-414a-bdec-2c15f4066036.png)

The encoders are all identical in structure (yet they do not share weights). Each one is broken down into two sub-layers

![image](https://user-images.githubusercontent.com/122109969/214243471-ec3081d6-535d-4bce-82b8-b57620eaff1b.png)

The encoder’s inputs first flow through a self-attention layer – a layer that helps the encoder look at other words in the input sentence as it encodes a specific word. We’ll look closer at self-attention later in the post.

The outputs of the self-attention layer are fed to a feed-forward neural network. The exact same feed-forward network is independently applied to each position.

The decoder has both those layers, but between them is an attention layer that helps the decoder focus on relevant parts of the input sentence (similar what attention does in seq2seq models).

![image](https://user-images.githubusercontent.com/122109969/214243623-d2be4998-5f48-4724-8303-11eb41e20d4f.png)

The embedding only happens in the bottom-most encoder. The abstraction that is common to all the encoders is that they receive a list of vectors each of the size 512 – In the bottom encoder that would be the word embeddings, but in other encoders, it would be the output of the encoder that’s directly below. The size of this list is hyperparameter we can set – basically it would be the length of the longest sentence in our training dataset.

After embedding the words in our input sequence, each of them flows through each of the two layers of the encoder.

![image](https://user-images.githubusercontent.com/122109969/214243718-f459614d-3b75-4542-bf3e-4d52bf1228c9.png)

![image](https://user-images.githubusercontent.com/122109969/214243803-6a35e1e5-a0e7-4a72-9e24-945ee95023b6.png)

Self-Attention at a High Level
Don’t be fooled by me throwing around the word “self-attention” like it’s a concept everyone should be familiar with. I had personally never came across the concept until reading the Attention is All You Need paper. Let us distill how it works.

Say the following sentence is an input sentence we want to translate:

”The animal didn't cross the street because it was too tired”
If you’re familiar with RNNs, think of how maintaining a hidden state allows an RNN to incorporate its representation of previous words/vectors it has processed with the current one it’s processing. Self-attention is the method the Transformer uses to bake the “understanding” of other relevant words into the one we’re currently processing.

![image](https://user-images.githubusercontent.com/122109969/214244091-5402f459-9fd9-414f-813d-31841e48f94c.png)

![image](https://user-images.githubusercontent.com/122109969/214244242-9b58eaae-0c64-4f64-a344-dd8c7c3e3955.png)
![image](https://user-images.githubusercontent.com/122109969/214244303-d4c39786-da40-4f18-827c-dbee7a3d8c2f.png)

![image](https://user-images.githubusercontent.com/122109969/214244342-b2a4ecc7-9090-4d46-8405-928156443461.png)

![image](https://user-images.githubusercontent.com/122109969/214244413-2e905e17-8237-450d-adde-4b985fbe0c4e.png)

Matrix Calculation of Self-Attention
The first step is to calculate the Query, Key, and Value matrices. We do that by packing our embeddings into a matrix X, and multiplying it by the weight matrices we’ve trained (WQ, WK, WV).

![image](https://user-images.githubusercontent.com/122109969/214244582-2b4a9e8f-0d08-4213-910e-f04a7c08b995.png)

Finally, since we’re dealing with matrices, we can condense steps two through six in one formula to calculate the outputs of the self-attention layer.

![image](https://user-images.githubusercontent.com/122109969/214244658-5f81dc37-cd15-4c55-a469-56e15c5b8285.png)

![image](https://user-images.githubusercontent.com/122109969/214245530-5e44e5b5-feb6-45f8-be18-f0373f389bb7.png)

Just checkout-------research paper https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf

![image](https://user-images.githubusercontent.com/122109969/214245180-f6d7c3c6-21f5-4573-b67d-90c9bb023bc7.png)



![image](https://user-images.githubusercontent.com/122109969/214245629-6682611e-c1fd-4ad0-b777-8778c585b173.png)

If we do the same self-attention calculation we outlined above, just eight different times with different weight matrices, we end up with eight different Z matrices

![image](https://user-images.githubusercontent.com/122109969/214245723-84784066-41bd-494b-b9f2-f68b6796af44.png)

That’s pretty much all there is to multi-headed self-attention. It’s quite a handful of matrices, I realize. Let me try to put them all in one visual so we can look at them in one place

![image](https://user-images.githubusercontent.com/122109969/214245803-b4609f96-5294-45e6-9326-402f96502d88.png)

Representing The Order of The Sequence Using Positional Encoding
One thing that’s missing from the model as we have described it so far is a way to account for the order of the words in the input sequence.

To address this, the transformer adds a vector to each input embedding. These vectors follow a specific pattern that the model learns, which helps it determine the position of each word, or the distance between different words in the sequence. The intuition here is that adding these values to the embeddings provides meaningful distances between the embedding vectors once they’re projected into Q/K/V vectors and during dot-product attention.

![image](https://user-images.githubusercontent.com/122109969/214246034-3a106587-3fe2-407e-a962-617511c31628.png)

![image](https://user-images.githubusercontent.com/122109969/214246082-1131bda6-262f-463d-af48-005d17f6bad3.png)

The Residuals
One detail in the architecture of the encoder that we need to mention before moving on, is that each sub-layer (self-attention, ffnn) in each encoder has a residual connection around it, and is followed by a layer-normalization step.

![image](https://user-images.githubusercontent.com/122109969/214246202-cc059e6f-7fee-4ad3-bd18-6fafd9271da2.png)

If we’re to visualize the vectors and the layer-norm operation associated with self attention, it would look like this:

![image](https://user-images.githubusercontent.com/122109969/214246317-074b9c5e-5586-4dab-866f-a0b9a1d17cf1.png)

This goes for the sub-layers of the decoder as well. If we’re to think of a Transformer of 2 stacked encoders and decoders, it would look something like this:

![image](https://user-images.githubusercontent.com/122109969/214246443-345931f5-f60c-438e-ace5-fe88e70149a5.png)

The Decoder Side
Now that we’ve covered most of the concepts on the encoder side, we basically know how the components of decoders work as well. But let’s take a look at how they work together.

The encoder start by processing the input sequence. The output of the top encoder is then transformed into a set of attention vectors K and V. These are to be used by each decoder in its “encoder-decoder attention” layer which helps the decoder focus on appropriate places in the input sequence:

![image](https://user-images.githubusercontent.com/122109969/214246551-5d9a0200-968d-4e2c-86f2-1135bc389689.png)

The following steps repeat the process until a special symbol is reached indicating the transformer decoder has completed its output. The output of each step is fed to the bottom decoder in the next time step, and the decoders bubble up their decoding results just like the encoders did. And just like we did with the encoder inputs, we embed and add positional encoding to those decoder inputs to indicate the position of each word.

![image](https://user-images.githubusercontent.com/122109969/214246755-93138be0-11da-4658-83da-3ac0a024e95c.png)

The Final Linear and Softmax Layer
The decoder stack outputs a vector of floats. How do we turn that into a word? That’s the job of the final Linear layer which is followed by a Softmax Layer.

The Linear layer is a simple fully connected neural network that projects the vector produced by the stack of decoders, into a much, much larger vector called a logits vector.

Let’s assume that our model knows 10,000 unique English words (our model’s “output vocabulary”) that it’s learned from its training dataset. This would make the logits vector 10,000 cells wide – each cell corresponding to the score of a unique word. That is how we interpret the output of the model followed by the Linear layer.

The softmax layer then turns those scores into probabilities (all positive, all add up to 1.0). The cell with the highest probability is chosen, and the word associated with it is produced as the output for this time step.

![image](https://user-images.githubusercontent.com/122109969/214246948-835c4a8a-5e22-4a8f-947e-5e5bd860e95c.png)


