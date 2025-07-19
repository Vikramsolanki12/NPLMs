# 🧠 NPLMs - Neural Probabilistic Language Models

An implementation of Neural Probabilistic Language Models (NPLMs) in Python using PyTorch. This project demonstrates how neural networks can be used to model the probability distribution of words and their contexts, as introduced by Bengio et al. (2003).

---

## 📌 What is an NPLM?

Neural Probabilistic Language Models learn continuous word representations and use a feedforward neural network to predict the probability of a word given its previous context (n-gram). It was one of the earliest models to introduce the concept of word embeddings and paved the way for modern deep NLP models.

---

## 🚀 Features

- Clean and modular NPLM architecture using PyTorch  
- Trains on n-gram-based context windows  
- Learns word embeddings during training  
- Cross-entropy loss minimization  
- Perplexity metric for evaluation  
- Supports custom datasets  

---

## 📊 Sample Results

| Epoch | Loss   | Perplexity |
|-------|--------|------------|
| 1     | 5.324  | 204.6      |
| 10    | 2.789  | 16.3       |
| 20    | 2.129  | 8.4        |

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/NPLMs.git
cd NPLMs
pip install -r requirements.txt
```

---

## 📚 References

- Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). *A Neural Probabilistic Language Model*. Journal of Machine Learning Research.  
- Mikolov, T. et al. (2013). *Efficient Estimation of Word Representations in Vector Space*.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## 📧 Contact

For questions or feedback, reach out at: **solankijogaram5@gmail.com**

---

## 📜 License
Copyright (c) 2025 Vikram Solanki

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell    
copies of the Software, and to permit persons to whom the Software is        
furnished to do so, subject to the following conditions:                     

The above copyright notice and this permission notice shall be included in   
all copies or substantial portions of the Software.                          

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR   
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,     
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE  
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER       
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING      
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS 
IN THE SOFTWARE.




