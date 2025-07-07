import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os


## The model is built according to inference.ipynb
## Please refer to that file.


# # Get sequence tower and move to correct device
# seq_tower = model.get_seq_tower()
# seq_tower.eval()


    # def encode_seqs(self, seqs, seq_attention_mask):
    #     seq_features = self.get_seq_tower()(seqs, seq_attention_mask)
    #     # the data type match
    #     seq_features = seq_features.to(dtype=self.get_model().mm_seq_projector.weight.dtype).to(self.device)
        
    #     seq_features = self.get_model().mm_seq_projector(seq_features)
    #     return seq_features


# Forward through encoder + projector to get projected embeddings
with torch.no_grad():
    # raw_seq_embeds = model.get_seq_tower()(seq_input_id, attention_mask=seq_attention_mask).squeeze(0).cpu()  # [L, D]
    # projected_seq_embeds = model.get_model().mm_seq_projector(raw_seq_embeds)  # [1, L, D]
    # seq_embeds = projected_seq_embeds.squeeze(0).cpu()
    seq_embeds =model.encode_seqs(seq_input_id, seq_attention_mask=seq_attention_mask).squeeze(0).cpu()

# # Remove batch dim
# seq_embeds = seq_embeds.squeeze(0).cpu()  # [L, D]

# Dimensionality reduction
pca = PCA(n_components=2)
seq_embeds_2d = pca.fit_transform(seq_embeds)

# Plot and save
plt.figure(figsize=(8, 6))
plt.scatter(seq_embeds_2d[:, 0], seq_embeds_2d[:, 1], c=range(seq_embeds_2d.shape[0]), cmap='viridis', s=20)
plt.colorbar(label="Amino Acid Position")
plt.title("Protein Sequence Embedding (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")

out_path = "asset/imgs/seq_projected_embedding_vis_pca.png"
plt.savefig(out_path, dpi=300)
print(f"✅ Embedding figure saved to: {os.path.abspath(out_path)}")


############### visualize in global language space

# 1. Get token embedding matrix (e.g., 128K x 4096)
token_embeds = model.model.embed_tokens.weight.detach().cpu()

# 2. Get sequence embeddings from your protein
seq_embeds =model.encode_seqs(seq_input_id, seq_attention_mask=seq_attention_mask).squeeze(0).cpu()

# 3. Combine them
combined = torch.cat([token_embeds, seq_embeds], dim=0)

# 4. PCA to 2D
pca = PCA(n_components=2)
combined_2d = pca.fit_transform(combined)

# 5. Split
n_vocab = token_embeds.shape[0]
token_2d = combined_2d[:n_vocab]
seq_2d = combined_2d[n_vocab:]

# 6. Plot
plt.figure(figsize=(10, 8))
plt.scatter(token_2d[:, 0], token_2d[:, 1], s=1, alpha=0.2, label="LLaMA3 Token Embeddings")
plt.scatter(seq_2d[:, 0], seq_2d[:, 1], c=range(len(seq_2d)), cmap="viridis", s=30, label="Protein Sequence Embedding")
plt.colorbar(label="Amino Acid Position")
plt.legend()
plt.title("Protein Sequence Embeddings in LLaMA3 Token Embedding Space")
plt.savefig("asset/imgs/embedding_in_llama3_token_space.png", dpi=300)
print("✅ Saved to embedding_in_llama3_token_space.png")
