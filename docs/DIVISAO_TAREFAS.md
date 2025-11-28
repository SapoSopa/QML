# Divisão de Tarefas - Projeto QML

## 🎯 Resumo da Divisão

Este documento detalha a distribuição de responsabilidades entre os 4 membros do projeto.

---

## 📊 Tabela Resumida

| Membro | Notebook | Trabalho Principal | Apresentação |
|--------|----------|-------------------|--------------|
| 1 🟥 | `01_conceitos_base_motivacao.ipynb` | Teoria + XOR | Motivação + conceitos + XOR |
| 2 🟦 | `02_dataset_circuito.ipynb` | Make_moons + embedding + circuito | Dataset + construção do VQC |
| 3 🟩 | `03_treinamento_otimizacao.ipynb` | Treinamento + gradients | Loss, otimização, como modelo aprende |
| 4 🟨 | `04_resultados_analise.ipynb` | Resultados + crítica | Fronteira de decisão + acurácia + limitações |

---

## 🟥 Membro 1 — Conceitos-Base e Motivação

### O que PRODUZIR no trabalho:

1. **Pesquisa e documentação de conceitos**:
   - O que são modelos variacionais
   - O que é embedding (alto nível)
   - O que são parâmetros ajustáveis em circuitos
   - O que são quantum gradients (parameter-shift rule)

2. **Dataset XOR**:
   - Código para gerar XOR
   - Visualização (scatter plot)
   - Explicação da não-linearidade

3. **Slides**:
   - 1 slide com XOR
   - Slides de motivação
   - Ponte para make_moons

### O que APRESENTAR:

- Por que ML quântico existe
- O que é um Variational Quantum Classifier
- O que é quantum gradients (sem matemática pesada)
- Dataset XOR como exemplo didático
- Transição para make_moons

### Tempo: 3 minutos

---

## 🟦 Membro 2 — Dataset e Construção do Circuito

### O que PRODUZIR no trabalho:

1. **Dataset make_moons**:
   - Geração com scikit-learn
   - Normalização (StandardScaler)
   - Split treino/teste
   - Visualização

2. **Embedding**:
   - Implementação de angle embedding
   - Documentação do processo

3. **Ansatz variacional**:
   - Circuito com 2 qubits
   - 2 layers de rotações + entanglement
   - Justificativa do design

4. **Visualização**:
   - Diagrama do circuito (PennyLane drawer)
   - Salvar figura

5. **Dados processados**:
   - Salvar em `data/` para próximos notebooks

### O que APRESENTAR:

- O que é make_moons e por que é não-linear
- Como transformamos 2D → rotações nos qubits
- Quantos qubits (2)
- Mostrar circuito (ansatz)
- Justificar tamanho (evitar barren plateau)

### Tempo: 3-4 minutos

---

## 🟩 Membro 3 — Treinamento, Gradientes e Otimização

### O que PRODUZIR no trabalho:

1. **QNode**:
   - Definir device
   - Criar QNode com diff_method='parameter-shift'
   - Conectar circuito + measurement

2. **Função de previsão**:
   - Converter expectation value → classe

3. **Loss function**:
   - MSE com labels {-1, +1}
   - Justificativa da escolha

4. **Loop de treinamento**:
   - Inicialização de parâmetros
   - Otimizador (GradientDescentOptimizer)
   - Mini-batch training
   - Logging de loss por época

5. **Demonstração de gradientes**:
   - Mostrar que PennyLane usa parameter-shift
   - Exemplo de cálculo de gradiente

6. **Salvar resultados**:
   - Parâmetros treinados → `results/`
   - Loss history → `results/`

### O que APRESENTAR:

- Como QNode funciona (input → circuito → expectation value)
- Por que MSE com {-1, +1}
- Como otimizador ajusta θ usando gradients
- Código do treino (trecho)
- Gráfico loss × epochs

### Tempo: 3-4 minutos

---

## 🟨 Membro 4 — Resultados, Fronteiras de Decisão e Análise Crítica

### O que PRODUZIR no trabalho:

1. **Avaliação do modelo**:
   - Carregar parâmetros treinados
   - Calcular acurácia no teste
   - Matriz de confusão
   - Classification report

2. **Fronteira de decisão**:
   - Criar grid de pontos
   - Fazer previsões no grid
   - Plotar contourf + contour
   - Overlay com dados de treino/teste

3. **Análise crítica**:
   - Escalabilidade (2 features → problema real)
   - Ruído quântico (simulador vs. hardware)
   - Sensibilidade ao ansatz
   - Comparação com ML clássico (SVM)

4. **Conclusões**:
   - O que funcionou
   - Limitações práticas
   - Vantagem quântica (realista)

### O que APRESENTAR:

- Fronteira de decisão aprendida
- Acurácia final
- Onde o modelo funciona/falha
- **Crítica honesta**:
  - "Isso escala mal para dados reais"
  - "Embedding é gargalo"
  - "Noise mataria o modelo no hardware atual"

### Tempo: 3-4 minutos

---

## 🔗 Dependências Entre Blocos

### Independente:
- **Bloco 1** (pode trabalhar sozinho)

### Sequencial:
- **Bloco 2** → gera dados → **Bloco 3** → treina modelo → **Bloco 4**

### Comunicação necessária:
- Membro 2 ↔ Membro 3: formato dos dados salvos
- Membro 3 ↔ Membro 4: formato dos parâmetros salvos
- Todos: definir número de qubits, layers, etc.

---

## 📋 Checklist por Membro

### Membro 1:
- [ ] Conceitos teóricos explicados
- [ ] XOR implementado e visualizado
- [ ] Slides de motivação prontos
- [ ] Ponte para make_moons

### Membro 2:
- [ ] Make_moons gerado e salvo
- [ ] Embedding implementado
- [ ] Ansatz construído
- [ ] Circuito visualizado
- [ ] Dados em `data/`

### Membro 3:
- [ ] QNode funcionando
- [ ] Loop de treinamento completo
- [ ] Gráfico de loss gerado
- [ ] Parâmetros salvos em `results/`
- [ ] Demo de parameter-shift

### Membro 4:
- [ ] Acurácia calculada
- [ ] Fronteira de decisão plotada
- [ ] Matriz de confusão
- [ ] Análise crítica completa
- [ ] Conclusões honestas

---

## 🎯 Pontos Críticos

### Para todos:
1. **Comunicação**: definir padrões de dados cedo
2. **Documentação**: comentar código generosamente
3. **Tempo**: ensaiar apresentação (3-4 min cada)
4. **Honestidade**: análise crítica é fundamental

### Dicas técnicas:
- Usar labels {-1, +1} (não {0, 1})
- Normalizar dados sempre
- Ansatz pequeno (evitar barren plateau)
- Salvar figuras em alta resolução

---

## 📅 Timeline Sugerido

1. **Semana 1**: Membros 1 e 2 (paralelo)
2. **Semana 2**: Membro 3 (após Membro 2)
3. **Semana 3**: Membro 4 (após Membro 3)
4. **Semana 4**: Integração + ensaio de apresentação

---

**Trabalhem de forma independente mas comuniquem-se regularmente!**
