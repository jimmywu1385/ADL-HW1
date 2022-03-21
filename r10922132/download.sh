mkdir -p cache/intent/
mkdir -p cache/slot/

mkdir -p ckpt/intent/
mkdir -p ckpt/slot/

wget https://www.dropbox.com/s/7tzpe6652ubv27r/intentbest.pt?dl=1 -O ckpt/intent/best.pt
wget https://www.dropbox.com/s/igc9xqo5ng5j421/slotbest.pt?dl=1 -O ckpt/slot/best.pt

wget https://www.dropbox.com/s/6d66cgi6b0af7k6/intentembeddings.pt?dl=1 -O cache/intent/embeddings.pt
wget https://www.dropbox.com/s/t461jdq1pis775d/slotembeddings.pt?dl=1 -O cache/slot/embeddings.pt

wget https://www.dropbox.com/s/nacbazdbe489tex/intent2idx.json?dl=1 -O cache/intent/intent2idx.json
wget https://www.dropbox.com/s/wp4yc4qdh8b9uz9/tag2idx.json?dl=1 -O cache/slot/tag2idx.json

wget https://www.dropbox.com/s/3jp0we6eqr0vcdl/intentvocab.pkl?dl=1 -O cache/intent/vocab.pkl
wget https://www.dropbox.com/s/wdu2pj4fph4qsjk/slotvocab.pkl?dl=1 -O cache/slot/vocab.pkl
