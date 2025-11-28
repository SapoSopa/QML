# 🌙 Quantum Machine Learning: Variational Quantum Classifier

Projeto de implementação de um **Variational Quantum Classifier (VQC)** usando PennyLane para classificação do dataset **make_moons**.

## 📋 Visão Geral

Este projeto demonstra a aplicação de computação quântica em machine learning, focando em:
- **Dataset principal**: Make Moons (classificação binária não-linear)
- **Dataset introdutório**: XOR (problema didático)
- **Modelo**: Variational Quantum Classifier (VQC)
- **Método de otimização**: Quantum gradients (parameter-shift rule)
- **Framework**: PennyLane

---

## 🎯 Estrutura do Projeto

```
QML/
├── notebooks/              # Notebooks Jupyter por bloco
│   ├── 01_conceitos_base_motivacao.ipynb
│   ├── 02_dataset_circuito.ipynb
│   ├── 03_treinamento_otimizacao.ipynb
│   └── 04_resultados_analise.ipynb
├── src/                    # Código Python reutilizável
│   ├── __init__.py
│   └── utils.py
├── data/                   # Dados processados (gerados pelos notebooks)
├── results/                # Resultados e gráficos
├── docs/                   # Documentação adicional
├── requirements.txt        # Dependências do projeto
└── README.md              # Este arquivo
```

---

## 👥 Divisão de Trabalho (4 Membros)

### 🟥 Membro 1 — Conceitos-Base e Motivação
**Notebook**: `01_conceitos_base_motivacao.ipynb`  
**Tempo de apresentação**: 3 minutos

**Responsabilidades:**
- Pesquisar e explicar conceitos fundamentais:
  - Modelos variacionais quânticos
  - Embedding de dados clássicos
  - Parâmetros ajustáveis em circuitos
  - Quantum gradients (parameter-shift rule)
- Criar visualização do dataset XOR
- Preparar slides de motivação
- Fazer ponte para o dataset make_moons

**Entregáveis:**
- [ ] Conceitos teóricos documentados
- [ ] Visualização do XOR
- [ ] Slides de introdução
- [ ] Seção do notebook completa

---

### 🟦 Membro 2 — Dataset e Construção do Circuito
**Notebook**: `02_dataset_circuito.ipynb`  
**Tempo de apresentação**: 3-4 minutos

**Responsabilidades:**
- Gerar e visualizar o dataset make_moons
- Normalizar dados (StandardScaler)
- Implementar angle embedding
- Desenvolver o ansatz variacional (2 qubits, 2 layers)
- Produzir diagrama do circuito quântico

**Entregáveis:**
- [ ] Dataset make_moons gerado e salvo
- [ ] Código de embedding funcionando
- [ ] Ansatz implementado
- [ ] Visualização do circuito
- [ ] Dados salvos em `data/`

---

### 🟩 Membro 3 — Treinamento, Gradientes e Otimização
**Notebook**: `03_treinamento_otimizacao.ipynb`  
**Tempo de apresentação**: 3-4 minutos

**Responsabilidades:**
- Implementar QNode (circuito + measurement)
- Criar função de previsão
- Definir loss function (MSE com labels {-1, +1})
- Desenvolver loop de treinamento
- Demonstrar parameter-shift no PennyLane
- Capturar e plotar loss × epochs

**Entregáveis:**
- [ ] QNode funcionando
- [ ] Loop de treinamento completo
- [ ] Gráfico de loss
- [ ] Parâmetros treinados salvos em `results/`
- [ ] Demonstração de gradientes quânticos

---

### 🟨 Membro 4 — Resultados, Fronteiras de Decisão e Análise Crítica
**Notebook**: `04_resultados_analise.ipynb`  
**Tempo de apresentação**: 3-4 minutos

**Responsabilidades:**
- Calcular acurácia no conjunto de teste
- Gerar fronteira de decisão (scatter + contour)
- Criar matriz de confusão
- **Análise crítica realista**:
  - Limitações de escalabilidade
  - Impacto do ruído quântico (NISQ)
  - Sensibilidade ao ansatz
  - Comparação honesta com ML clássico

**Entregáveis:**
- [ ] Acurácia e métricas calculadas
- [ ] Fronteira de decisão visualizada
- [ ] Matriz de confusão
- [ ] Análise crítica completa
- [ ] Discussão sobre vantagem quântica

---

## 🚀 Setup e Instalação

### 1. Clone o repositório (ou crie o ambiente)
```bash
cd /home/saposopa/Saparia/QML
```

### 2. Crie um ambiente virtual
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Inicie o Jupyter Notebook
```bash
jupyter notebook
```

---

## 📊 Workflow de Execução

Os notebooks devem ser executados **em ordem**:

1. **Notebook 1**: Conceitos e XOR (independente)
2. **Notebook 2**: Gera dados → salva em `data/`
3. **Notebook 3**: Carrega dados → treina modelo → salva em `results/`
4. **Notebook 4**: Carrega modelo → avalia → análise crítica

---

## 🔬 Tecnologias Utilizadas

- **PennyLane**: Framework de computação quântica
- **NumPy**: Computação numérica
- **Scikit-learn**: Dataset e métricas
- **Matplotlib/Seaborn**: Visualizações
- **Jupyter**: Ambiente interativo

---

## 📈 Resultados Esperados

### O que deve funcionar:
✅ VQC aprende problema não-linear (make_moons)  
✅ Parameter-shift calcula gradientes quânticos  
✅ Fronteira de decisão captura não-linearidade  
✅ Acurácia razoável (~80-90% esperado)

### Limitações reconhecidas:
❌ Não escala para datasets reais (>2 features)  
❌ Ruído quântico impede uso em hardware atual  
❌ Nenhuma vantagem demonstrada vs. SVM clássico  
❌ Sensível ao design do ansatz (trial-and-error)

---

## 🎓 Apresentação Final

### Tempo total: ~12-15 minutos

| Bloco | Tempo | Foco |
|-------|-------|------|
| 1 🟥  | 3 min | Motivação + Conceitos + XOR |
| 2 🟦  | 3-4 min | Dataset + Circuito |
| 3 🟩  | 3-4 min | Treinamento + Gradientes |
| 4 🟨  | 3-4 min | Resultados + Crítica |

**Cada membro apresenta seu próprio bloco de forma independente.**

---

## 📚 Referências

- [PennyLane Documentation](https://pennylane.ai/)
- [Variational Quantum Algorithms](https://arxiv.org/abs/2012.09265)
- [Parameter-shift rules](https://pennylane.ai/qml/glossary/parameter_shift.html)
- [Barren Plateaus in QML](https://arxiv.org/abs/1803.11173)

---

## ⚠️ Notas Importantes

### Para os membros:
1. **Notebooks são independentes**: cada um pode trabalhar em paralelo
2. **Comunicação é essencial**: definir formato de dados salvos
3. **Análise crítica honesta**: não vender hype, mostrar realidade
4. **Código limpo**: comentar bem, usar funções reutilizáveis

### Pontos de atenção:
- Normalização dos dados é **crítica** para convergência
- Labels devem ser {-1, +1} (não {0, 1}) para PauliZ
- Ansatz pequeno evita barren plateaus
- Tempo de execução: treino pode levar 5-10 minutos

---

## 📞 Contato e Suporte

Para dúvidas sobre:
- **Bloco 1**: [Membro 1]
- **Bloco 2**: [Membro 2]
- **Bloco 3**: [Membro 3]
- **Bloco 4**: [Membro 4]

---

## 🏁 Checklist Final do Projeto

- [ ] Todos os notebooks executam sem erros
- [ ] Dados salvos em `data/`
- [ ] Modelo treinado salvo em `results/`
- [ ] Todas as visualizações geradas
- [ ] Slides de apresentação preparados
- [ ] Análise crítica completa
- [ ] Tempo de apresentação ensaiado

---

**Boa sorte com o projeto! 🚀🔬**
