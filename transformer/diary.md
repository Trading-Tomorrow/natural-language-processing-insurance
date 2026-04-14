## Architecture Overview

Nesta fase o projeto já tem duas camadas implementadas:

-> Um tokenizer BPE customizado para o domínio de sinistros automóvel
-> Um primeiro stack pairwise para classificação entre duas statements

O objetivo do stack pairwise é receber duas statements já tokenizadas no formato:

`[CLS] statement_a [SEP] statement_b [SEP]`

e produzir uma decisão de 3 classes:

-> `supports`
-> `neutral`
-> `contradicts`

Esta arquitetura ainda não faz agregação ao nível do claim inteiro. O foco atual é deteção local de consistência ou contradição entre dois textos.

## Porque Escolhi um Setup Pairwise

No problema de fraude e inconsistência em seguros, muitas anomalias não aparecem numa statement isolada. Elas emergem quando duas versões do mesmo evento são comparadas.

Por isso, a formulação pairwise faz sentido como primeira etapa:

-> A contradição aparece entre statement A e statement B, não apenas dentro de uma statement sozinha
-> O modelo fica mais interpretável porque a unidade de decisão passa a ser um par concreto de textos
-> Um único claim pode gerar vários exemplos de treino, por exemplo insured_driver vs third_party_driver, insured_driver vs witness, third_party_driver vs witness
-> Isto é uma etapa mais controlada antes de construir uma camada posterior de agregação ao nível do claim completo

Importante: esta hipótese de modelação já está implementada no código através de um builder weakly supervised. O projeto já consegue transformar os claims existentes em pares `supports`, `neutral` e `contradicts`, embora estes labels ainda não sejam uma anotação manual gold.

## O Que o Tokenizer Já Faz

O tokenizer já estava concluído antes desta etapa.

Ele faz:

-> Tokenização BPE customizada, treinada com corpus do domínio
-> Uso de tokens especiais do domínio como `<insured_driver>`, `<third_party_driver>`, `<insurance_adjuster>`, `<speed>`, `<vehicle>`, `<claim>`, `<sep_stmt>`, entre outros
-> Normalização de velocidade antes da tokenização, por exemplo:
`10 km/h -> <speed> 10 kmh`
-> Conversão de texto para tokens e depois para token IDs

É importante notar que nesta fase o tokenizer ainda não faz entendimento semântico. Ele apenas converte texto em unidades discretas manipuláveis pelo modelo neural. O entendimento de relação entre statements fica para o transformer encoder.

## Formato de Entrada do Modelo

O dataset pairwise constrói explicitamente a sequência final como:

`[CLS] statement_a [SEP] statement_b [SEP]`

O significado dos tokens especiais é:

-> `[CLS]` marca o início da sequência completa e serve como posição principal de pooling quando usamos a estratégia `cls`
-> `[SEP]` separa statement A de statement B e também fecha a sequência no final
-> `[PAD]` é usado para completar a sequência até `max_length=256`

No código atual:

-> `statement_a` e `statement_b` são tokenizadas sem adicionar special tokens automaticamente
-> O dataset reserva 3 posições para `[CLS]`, primeiro `[SEP]` e último `[SEP]`
-> Se o par for demasiado grande, o truncation é feito removendo tokens do lado mais longo até caber no limite

## O Que Cada Tensor Representa

O `dataset.py` devolve quatro tensores principais:

-> `input_ids`: sequência final de IDs já com `[CLS] statement_a [SEP] statement_b [SEP] [PAD] ...`
-> `attention_mask`: máscara binária onde `1` significa token válido e `0` significa padding
-> `token_type_ids`: indicador de segmento
-> `labels`: classe supervisionada do par

No caso de `token_type_ids`, a implementação atual usa:

-> segmento `0` para `[CLS] statement_a [SEP]`
-> segmento `1` para `statement_b [SEP]`
-> tokens `[PAD]` continuam com valor `0`

As labels seguem exatamente este mapeamento:

-> `supports = 0`
-> `neutral = 1`
-> `contradicts = 2`

## Arquitetura Pairwise Implementada

O `model.py` implementa um transformer encoder do zero em PyTorch, sem usar classes de modelo do Hugging Face.

### Embeddings

O modelo soma três embeddings:

-> Token embeddings, aprendidos a partir dos `input_ids`
-> Positional embeddings, também aprendidos, para representar a ordem dos tokens
-> Segment embeddings, opcionais mas ativas por defeito, para distinguir statement A de statement B através de `token_type_ids`

Depois da soma, a implementação aplica `LayerNorm` e `Dropout`.

### Transformer Encoder

O encoder é composto por uma stack de blocos `TransformerEncoderBlock`.

Cada bloco tem:

-> `nn.MultiheadAttention` com `batch_first=True`
-> Ligação residual
-> `LayerNorm` em esquema pre-norm
-> Feed-forward network com projeção linear, ativação `GELU`, dropout e projeção de volta ao espaço oculto

A máscara de atenção é construída a partir de `attention_mask`, usando `key_padding_mask=(attention_mask == 0)` para que o self-attention ignore padding.

### Pooling

A arquitetura suporta duas estratégias:

-> `cls`, que usa o vetor oculto da posição 0
-> `mean`, que faz média apenas sobre tokens não-padding

A configuração por defeito no código é `pooling="cls"`.

### Classifier Head

Depois do pooling, o modelo aplica:

-> `Dropout`
-> `Linear(hidden_size -> 3)`

O `forward()` devolve:

-> `logits`
-> `probs`
-> `loss` opcional quando `labels` é passado

Atualmente a loss usada é `cross_entropy`, apropriada para classificação multiclasse com 3 rótulos mutuamente exclusivos.

## Ficheiros Implementados

### `model.py`

Implementa:

-> `PairwiseTransformerConfig`
-> `PairwiseEmbeddings`
-> `TransformerEncoderBlock`
-> `PairwiseTransformerClassifier`

Os defaults atuais da baseline são:

-> `max_position_embeddings = 256`
-> `hidden_size = 256`
-> `num_hidden_layers = 4`
-> `num_attention_heads = 8`
-> `intermediate_size = 1024`
-> `dropout = 0.1`
-> `attention_dropout = 0.1`
-> `type_vocab_size = 2`
-> `num_labels = 3`
-> `pooling = "cls"`
-> `use_segment_embeddings = True`

### `pairwise_utils.py`

Implementa:

-> mapeamento de labels para IDs
-> mapeamento inverso de IDs para labels
-> função `build_token_type_ids()` para separar statement A e statement B

### `dataset.py`

Implementa:

-> leitura de datasets `.json` e `.jsonl`
-> validação da estrutura esperada:
`pair_id`, `text_a`, `text_b`, `label`
-> tokenização com o tokenizer já treinado em `transformer/tokenizers/claims_bpe`
-> montagem manual da sequência pairwise
-> truncation pair-aware
-> devolução de tensores PyTorch prontos para treino ou inferência

### `test_model.py`

Implementa um sanity check mínimo:

-> cria um pequeno dataset sintético temporário
-> carrega o tokenizer real do projeto
-> instancia o dataset pairwise
-> monta um batch
-> verifica shapes dos tensores
-> verifica `token_type_ids`
-> faz um `forward()` completo
-> confirma que a loss é calculada corretamente
-> testa também pooling por média

### `build_pairwise_dataset.py`

Implementa:

-> carregamento do corpus limpo através de `dataset_cleaning.py`
-> geração de pares within-claim entre `insured_driver` e outra statement do mesmo claim
-> geração de pares `neutral` entre claims diferentes
-> escrita de dois ficheiros:
`transformer/data/pairwise_dataset.jsonl`
e
`transformer/data/pairwise_dataset_full.jsonl`
-> escrita de estatísticas em
`transformer/data/pairwise_dataset_stats.json`

O builder atual usa weak supervision:

-> claims `genuine_accident` geram pares `supports`
-> claims fraudulentos podem gerar `supports` ou `contradicts` consoante a heurística baseada em `fraud_indicators`
-> pares `neutral` são amostrados entre statements de claims diferentes

### `train_pairwise.py`

Implementa:

-> treino supervisionado do encoder pairwise sobre `pairwise_dataset.jsonl` por defeito
-> split estratificado train/validation com `validation_ratio = 0.2`
-> criação de `DataLoader`
-> otimização com `AdamW`
-> suporte opcional a `class weighting`
-> cálculo de `loss`, `accuracy`, `macro_precision`, `macro_recall` e `macro_f1`
-> seleção do melhor checkpoint com base em `validation macro_f1`
-> escrita de `best_model.pt` e `training_history.json`

O script atual é suficientemente modular para:

-> trocar dataset por argumento de linha de comando
-> treinar com `pooling = cls` ou `mean`
-> alterar largura, profundidade e dropout da baseline sem mudar o código do modelo
-> usar `--class-weighting balanced` para aplicar pesos por classe na `cross_entropy`

Quando `class weighting` está ativo, o treino usa a fórmula:

-> `weight_c = N / (C * n_c)`

onde:

-> `N` é o número total de exemplos do split de treino
-> `C` é o número de classes
-> `n_c` é o número de exemplos da classe `c`

Isto permite treinar sobre `pairwise_dataset_full.jsonl` sem deitar fora volume de treino por downsampling agressivo.

### `evaluate_pairwise.py`

Implementa:

-> carregamento de um checkpoint guardado por `train_pairwise.py`
-> reconstrução determinística do split train/validation através de `seed` e `validation_ratio` guardados no checkpoint
-> avaliação em `train`, `validation` ou `full`
-> reutilização dos `class_weights` guardados no checkpoint, quando existirem
-> cálculo de `loss`, `accuracy`, `macro_precision`, `macro_recall` e `macro_f1`
-> cálculo de métricas por classe
-> construção de matriz de confusão
-> escrita de um relatório JSON com os resultados

### `plot_training_history.py`

Implementa:

-> leitura de `training_history.json`
-> geração de curvas de treino e validação
-> visualização de `loss`, `accuracy` e `macro_f1`
-> marcação explícita da `best_epoch`
-> escrita do gráfico em PNG para inclusão no relatório ou no diário

## Dataset Pairwise Gerado

Depois da limpeza e deduplicação dos datasets base, o corpus final usado para gerar pares tem:

-> `3567` claims
-> `4170` claims originais antes da limpeza
-> `577` remoções por `claim_id` duplicado
-> `26` remoções por fingerprint textual duplicada

As fontes que compõem o corpus limpo atual são:

-> o corpus original misto
-> o corpus `good_only`
-> o novo corpus `mixed_diverse` com `1000` claims adicionais

Depois desta expansão, o builder pairwise passou a operar sobre um corpus mais rico em fraude e com mais variedade lexical.

Na versão atual do dataset pairwise:

-> `supports = 1506`
-> `neutral = 2376`
-> `contradicts = 870`
-> `full_examples = 4752`

Isto é importante porque um claim não gera automaticamente um exemplo pairwise. Para haver um par within-claim no builder atual, é necessário existir:

-> pelo menos uma statement `insured_driver`
-> pelo menos uma outra statement com role diferente

Na versão anterior do corpus, apenas `1358` claims satisfaziam esse critério mínimo de pareamento interno. Com a adição do novo dataset misto mais diverso, a disponibilidade de pares contraditórios aumentou bastante.

## Como o Dataset Balanceado Evoluiu com a Expansão do Corpus

O número de claims e o número de pares de treino não são a mesma coisa, mas a expansão do corpus alterou significativamente o bottleneck anterior.

Antes da expansão:

-> `supports = 1221`
-> `contradicts = 147`
-> dataset balanceado final = `441` pares

Depois da expansão com `mixed_diverse`:

-> `supports = 1506`
-> `contradicts = 870`
-> `neutral = 2376`

Como o builder continua a balancear por downsampling usando a menor classe, o novo alvo por classe passou para:

-> `870 supports`
-> `870 neutral`
-> `870 contradicts`

Resultado atual:

-> `2610` exemplos no ficheiro balanceado `pairwise_dataset.jsonl`

O ficheiro completo weakly supervised passou a ter:

-> `1506 supports`
-> `2376 neutral`
-> `870 contradicts`
-> `4752` exemplos em `pairwise_dataset_full.jsonl`

## Primeiro Resultado Experimental

Foi executado um primeiro treino supervisionado baseline sem class weighting e usando o dataset balanceado:

-> dataset: `transformer/data/pairwise_dataset.jsonl`
-> total de exemplos: `441`
-> split estratificado: `354` treino e `87` validação
-> `batch_size = 32`
-> `num_epochs = 32`
-> `hidden_size = 128`
-> `num_hidden_layers = 8`
-> `num_attention_heads = 8`
-> `intermediate_size = 256`
-> `learning_rate = 3e-4`
-> `pooling = cls`

O melhor checkpoint foi:

-> `best_epoch = 25`
-> `best_validation_macro_f1 = 0.8466`
-> `validation_accuracy = 0.8506`
-> `validation_loss = 0.6030`

O gráfico deste treino foi guardado em:

-> `transformer/figures/pairwise_baseline_balanced_training_curves.png`

Leitura técnica do comportamento observado:

-> nas primeiras épocas, treino e validação melhoram de forma consistente, o que indica que o modelo consegue aprender o padrão pairwise mesmo num corpus relativamente pequeno
-> a validação atinge o melhor ponto na `epoch 25`
-> depois da `epoch 25`, as métricas de treino continuam elevadas, mas as métricas de validação degradam-se e oscilam bastante
-> por exemplo, entre as `epochs 26` e `32`, o `validation macro_f1` cai de `0.8466` para valores na zona de `0.7335-0.8259`, e a `validation loss` chega a `1.2092` na `epoch 29`

Esta divergência entre treino e validação sugere:

-> início de overfitting depois do melhor ponto de validação
-> alguma instabilidade de otimização compatível com dataset pequeno e modelo relativamente profundo para o volume disponível
-> necessidade prática de selecionar o melhor checkpoint por validação, e não o último epoch

Em termos metodológicos, este resultado é suficientemente forte para validar a baseline pairwise:

-> o pipeline tokenizer -> dataset pairwise -> transformer encoder -> treino supervisionado está funcional
-> o modelo já consegue atingir performance útil no split de validação atual
-> a próxima melhoria deve focar protocolo experimental e qualidade de dataset, mais do que aumentar imediatamente a complexidade da arquitetura

## Suporte ao Dataset Completo com Class Weighting

Depois do primeiro treino no dataset balanceado, o código foi estendido para suportar treino no dataset completo:

-> `transformer/data/pairwise_dataset_full.jsonl`
-> `2736` exemplos
-> distribuição global:
`1221 supports`, `1368 neutral`, `147 contradicts`

O objetivo desta extensão é simples:

-> manter mais volume de treino
-> evitar descartar exemplos por balancing via downsampling
-> compensar o desbalanceamento através de pesos na loss

Num smoke test curto de validação da implementação, foi executado um treino de `1` época sobre o dataset completo com `--class-weighting balanced`.

Nesse run, o split de treino ficou com:

-> `977 supports`
-> `1094 neutral`
-> `118 contradicts`

Os pesos calculados automaticamente foram:

-> `supports = 0.7468`
-> `neutral = 0.6670`
-> `contradicts = 6.1836`

Este teste curto confirmou:

-> cálculo correto dos pesos por classe
-> passagem dos pesos para a `cross_entropy`
-> persistência dos pesos no checkpoint
-> reutilização dos mesmos pesos no script de avaliação

Importante: este smoke test de `1` época não é ainda um resultado experimental comparável com o treino de `32` épocas no dataset balanceado. Ele apenas valida que a nova variante de treino `full + class weighting` está operacional.

## Baseline de Referência Antes da Expansão

Depois da validação inicial da variante `full + class weighting`, foi executado o treino completo com a mesma configuração estrutural base:

-> dataset: `transformer/data/pairwise_dataset_full.jsonl`
-> total de exemplos: `2736`
-> split estratificado: `2189` treino e `547` validação
-> `class_weighting = balanced`
-> `batch_size = 32`
-> `num_epochs = 32`
-> `hidden_size = 128`
-> `num_hidden_layers = 8`
-> `num_attention_heads = 8`
-> `intermediate_size = 256`
-> `learning_rate = 3e-4`
-> `pooling = cls`

Os pesos usados na loss foram:

-> `supports = 0.7468`
-> `neutral = 0.6670`
-> `contradicts = 6.1836`

O melhor checkpoint deste run foi:

-> `best_epoch = 13`
-> `best_validation_macro_f1 = 0.8667`
-> `validation_accuracy = 0.8665`
-> `validation_loss = 0.4934`

O gráfico principal desta baseline foi guardado em:

-> `transformer/figures/pairwise_full_weighted_8_layers_training_curves.png`

O dashboard de avaliação desta baseline foi guardado em:

-> `transformer/figures/pairwise_full_weighted_8_layers_validation_dashboard.png`

Este resultado passa agora a ser a baseline principal do projeto por duas razões:

-> supera o treino anterior no dataset balanceado, que tinha `best_validation_macro_f1 = 0.8466`
-> usa mais volume de treino sem descartar exemplos por downsampling agressivo

Em termos absolutos, a melhoria sobre a baseline anterior foi:

-> `0.8667 - 0.8466 = +0.0201` em `validation macro_f1`

## Resultado com Corpus Expandido

Depois da integração do novo dataset `mixed_diverse`, o pipeline foi refeito:

-> limpeza e deduplicação do corpus atualizado
-> retreino do tokenizer
-> reconstrução do dataset pairwise
-> novo treino do modelo pairwise com a mesma arquitetura base

Configuração do primeiro treino expandido:

-> dataset: `transformer/data/pairwise_dataset_full.jsonl`
-> total de exemplos: `4752`
-> split estratificado: `3802` treino e `950` validação
-> `class_weighting = balanced`
-> `batch_size = 32`
-> `num_epochs = 32`
-> `hidden_size = 128`
-> `num_hidden_layers = 8`
-> `num_attention_heads = 8`
-> `intermediate_size = 256`
-> `learning_rate = 3e-4`
-> `pooling = cls`
-> `device = mps`

Os novos pesos usados na loss foram:

-> `supports = 1.0517`
-> `neutral = 0.6667`
-> `contradicts = 1.8209`

O melhor checkpoint deste primeiro run expandido foi:

-> `best_epoch = 21`
-> `best_validation_macro_f1 = 0.8575`
-> `validation_accuracy = 0.8611`
-> `validation_loss = 0.4927`

Os gráficos deste run foram guardados em:

-> `transformer/figures/pairwise_full_weighted_expanded_mps_training_curves.png`
-> `transformer/figures/pairwise_full_weighted_expanded_mps_validation_dashboard.png`

Comparação com a baseline de referência antes da expansão:

-> baseline anterior: `macro_f1 = 0.8667`
-> corpus expandido: `macro_f1 = 0.8575`
-> diferença absoluta: `-0.0092`

Este primeiro resultado expandido mostrou que adicionar mais dados não garantiu melhoria automática na métrica final. A leitura mais plausível foi:

-> o novo corpus tornou a tarefa mais difícil e semanticamente mais variada
-> a validação expandida tem mais exemplos e uma fronteira entre classes menos trivial
-> o modelo continua forte, mas já não beneficia apenas de volume; a qualidade e a natureza dos pares continuam a ser decisivas

## Sweep Arquitetural Pós-Expansão

Depois deste primeiro run expandido, foi executada uma pequena grelha de experiências para perceber se o gargalo estava em profundidade, largura ou capacidade do feed-forward network.

Resultados observados:

-> `L8 H128 I256`: `macro_f1 = 0.8575`
-> `L8 H128 I512`: `macro_f1 = 0.8694`
-> `L8 H192 I768`: `macro_f1 = 0.8569`
-> `L8 H256 I1024`: `macro_f1 = 0.8530`
-> `L10 H192 I768`: `macro_f1 = 0.8646`

Leitura técnica da sweep:

-> a melhor melhoria veio de aumentar `intermediate_size` de `256` para `512`, mantendo `hidden_size = 128` e `num_hidden_layers = 8`
-> aumentar largura para `192` ou `256` não ajudou
-> aumentar profundidade para `10` layers também não superou a melhor variante com `8` layers
-> isto sugere que, neste projeto, o bottleneck atual estava mais na capacidade do bloco feed-forward do que na profundidade global da stack

## Baseline Principal Antes das Ablações Pairwise

Antes das ablações seguintes, a baseline principal do projeto passou a ser:

-> dataset: `transformer/data/pairwise_dataset_full.jsonl`
-> total de exemplos: `4752`
-> split estratificado: `3802` treino e `950` validação
-> `class_weighting = balanced`
-> `batch_size = 32`
-> `num_epochs = 32`
-> `hidden_size = 128`
-> `num_hidden_layers = 8`
-> `num_attention_heads = 8`
-> `intermediate_size = 512`
-> `learning_rate = 3e-4`
-> `pooling = cls`
-> `device = mps`

O melhor checkpoint desta baseline foi:

-> `best_epoch = 24`
-> `best_validation_macro_f1 = 0.8694`
-> `validation_accuracy = 0.8695`
-> `validation_loss = 0.3903`

Os gráficos desta baseline foram guardados em:

-> `transformer/figures/pairwise_full_weighted_expanded_ffn512_training_curves.png`
-> `transformer/figures/pairwise_full_weighted_expanded_ffn512_validation_dashboard.png`

Comparações relevantes:

-> contra o primeiro run expandido `L8 H128 I256`: `0.8694 - 0.8575 = +0.0119`
-> contra a antiga baseline de referência pré-expansão: `0.8694 - 0.8667 = +0.0027`

Em termos práticos, isto significa:

-> o corpus expandido não melhorou automaticamente a baseline original
-> mas, depois de um ajuste arquitetural pequeno e bem direcionado, a nova baseline passou a superar todas as anteriores

## Análise de Acertos e Erros da Baseline Atual

Na validação da baseline atual, o modelo avaliou `950` pares.

Resultado global:

-> `826` acertos
-> `124` erros
-> `accuracy = 0.8695`
-> `macro_f1 = 0.8694`

Por classe, o número de acertos e erros foi:

-> `supports`: `277` acertos e `24` erros
-> `neutral`: `385` acertos e `90` erros
-> `contradicts`: `164` acertos e `10` erros

As métricas por classe mostram um padrão importante:

-> `supports`: `precision = 0.7937`, `recall = 0.9203`, `f1 = 0.8523`
-> `neutral`: `precision = 0.9601`, `recall = 0.8105`, `f1 = 0.8790`
-> `contradicts`: `precision = 0.8200`, `recall = 0.9425`, `f1 = 0.8770`

A matriz de confusão mostra onde o erro realmente acontece:

-> `supports -> supports`: `277`
-> `supports -> neutral`: `11`
-> `supports -> contradicts`: `13`
-> `neutral -> supports`: `67`
-> `neutral -> neutral`: `385`
-> `neutral -> contradicts`: `23`
-> `contradicts -> supports`: `5`
-> `contradicts -> neutral`: `5`
-> `contradicts -> contradicts`: `164`

Isto permite uma leitura mais concreta do que o modelo está a aprender:

-> o modelo continua a reconhecer bem `supports`, mas o ganho mais claro apareceu nas classes mais difíceis
-> `neutral` melhorou em `f1`, mantendo precisão muito alta
-> `contradicts` melhorou claramente em recall e em `f1`, o que indica maior sensibilidade a conflito narrativo sem colapso forte de precisão
-> o aumento do FFN parece ter ajudado precisamente na separação das fronteiras mais subtis

O erro dominante da baseline é:

-> `neutral -> supports` com `67` casos

O segundo padrão de erro mais importante é:

-> `neutral -> contradicts` com `23` casos

O terceiro padrão relevante passa a ser:

-> `supports -> contradicts` com `13` casos

Isto sugere que o principal limite atual não é distinguir exemplos claramente suportivos, mas sim separar com mais consistência:

-> suporte genuíno vs neutralidade
-> neutralidade vs contradição
-> suporte genuíno vs contradição mais subtil

Uma interpretação plausível, consistente com a forma como o dataset foi construído, é:

-> o novo corpus aumentou diversidade lexical e diversidade de cenários, reduzindo padrões fáceis de memorizar
-> a variante `ffn512` conseguiu aproveitar melhor esse corpus do que a variante base `i256`
-> a fronteira entre `neutral` e `contradicts` continua a ser a zona semanticamente mais delicada, mas a cobertura das classes ficou mais equilibrada

Esta leitura é importante para pesquisa porque mostra que já não estamos apenas perante uma métrica global. A análise de erros permite dizer, com base em evidência concreta, que o modelo:

-> já aprende relações pairwise semanticamente úteis
-> não está a prever classes raras ao acaso
-> continua a precisar de melhor separação nas fronteiras entre `supports`, `neutral` e `contradicts`

## Tentativa de Contextualização Explícita do Claim

Depois da baseline `ffn512`, foi implementada uma nova variante arquitetural com dois objetivos:

-> expor `incident_type` e `detected_damages` explicitamente no texto de entrada de cada lado do par
-> substituir o head puramente baseado em `[CLS]` por um head de comparação explícita entre os dois segmentos

Na prática, cada lado do par passou a ser codificado com contexto adicional do próprio claim, por exemplo:

-> `incident_type: ...`
-> `detected_damages: ...`
-> `vehicle: ...`
-> token de role
-> texto da statement

Além disso, o head final passou a usar a concatenação de:

-> `global_pool`
-> `pool_a`
-> `pool_b`
-> `|pool_a - pool_b|`
-> `pool_a * pool_b`

Esta mudança é conceptualmente consistente com o problema, porque a tarefa pairwise não depende apenas de representação global do par, mas também de comparação explícita entre os dois lados.

## Resultado da Variante Contextualizada

Foi então executado um treino completo desta nova variante com:

-> dataset: `transformer/data/pairwise_dataset_full.jsonl`
-> total de exemplos: `4752`
-> split estratificado: `3802` treino e `950` validação
-> `class_weighting = balanced`
-> `batch_size = 32`
-> `num_epochs = 32`
-> `hidden_size = 128`
-> `num_hidden_layers = 8`
-> `num_attention_heads = 8`
-> `intermediate_size = 512`
-> `learning_rate = 3e-4`
-> `pooling = cls`
-> `device = mps`
-> `use_pairwise_comparison_head = True`

O melhor checkpoint desta variante foi:

-> `best_epoch = 27`
-> `best_validation_macro_f1 = 0.8580`
-> `validation_accuracy = 0.8611`
-> `validation_loss = 0.5834`

Os gráficos desta variante foram guardados em:

-> `transformer/figures/pairwise_full_weighted_expanded_ffn512_contextual_training_curves.png`
-> `transformer/figures/pairwise_full_weighted_expanded_ffn512_contextual_validation_dashboard.png`

Comparação direta com a baseline principal `ffn512` sem esta contextualização explícita:

-> baseline principal: `macro_f1 = 0.8694`
-> variante contextualizada: `macro_f1 = 0.8580`
-> diferença absoluta: `-0.0114`

Ou seja:

-> a ideia arquitetural é plausível
-> mas, neste primeiro teste, ela não superou a baseline principal

## Leitura dos Erros da Variante Contextualizada

Na validação desta variante contextualizada, o modelo obteve:

-> `818` acertos
-> `132` erros
-> `accuracy = 0.8611`
-> `macro_f1 = 0.8580`

Métricas por classe:

-> `supports`: `precision = 0.8387`, `recall = 0.8638`, `f1 = 0.8511`
-> `neutral`: `precision = 0.9087`, `recall = 0.8379`, `f1 = 0.8719`
-> `contradicts`: `precision = 0.7921`, `recall = 0.9195`, `f1 = 0.8511`

Matriz de confusão:

-> `supports -> supports`: `260`
-> `supports -> neutral`: `28`
-> `supports -> contradicts`: `13`
-> `neutral -> supports`: `48`
-> `neutral -> neutral`: `398`
-> `neutral -> contradicts`: `29`
-> `contradicts -> supports`: `2`
-> `contradicts -> neutral`: `12`
-> `contradicts -> contradicts`: `160`

A leitura mais útil desta comparação é:

-> a variante contextualizada reduziu `neutral -> supports` de `67` para `48`
-> mas aumentou `supports -> neutral` de `11` para `28`
-> e aumentou `contradicts -> neutral` de `5` para `12`

Isto sugere que a contextualização explícita com `incident_type` e `detected_damages`, combinada com o comparison head, tornou o modelo menos propenso a colapsar tudo em `supports`, mas ao mesmo tempo empurrou mais casos para `neutral`.

Em termos metodológicos, o resultado é importante porque mostra:

-> a feature não foi inútil; ela alterou claramente a geometria de decisão do modelo
-> mas a forma atual de injetar contexto ainda não produz ganho líquido em `macro_f1`
-> a baseline principal continua a ser a variante `ffn512` sem esta contextualização explícita

## Ablação Entre Contexto Explícito e Comparison Head

Depois da implementação da variante contextualizada, a comparação correta deixou de ser apenas entre duas arquiteturas isoladas. Foi necessário separar dois efeitos diferentes:

-> presença ou ausência de contexto explícito do claim no texto de entrada
-> presença ou ausência de comparison head explícito no classificador final

Para isso, o builder pairwise passou a gerar duas versões do corpus:

-> `transformer/data/pairwise_dataset_full.jsonl` para a variante `plain`
-> `transformer/data/pairwise_dataset_full_contextual.jsonl` para a variante `contextual`

Foram então consolidadas quatro variantes relevantes:

-> `plain + no comparison head`: `macro_f1 = 0.8694`
-> `plain + comparison head`: `macro_f1 = 0.8705`
-> `contextual + no comparison head`: `macro_f1 = 0.8658`
-> `contextual + comparison head`: `macro_f1 = 0.8580`

## Resultado da Variante `Plain + Comparison Head`

Esta variante usa:

-> input `plain`, isto é, statements apenas com token de role e texto original
-> comparison head explícito
-> dataset: `transformer/data/pairwise_dataset_full.jsonl`
-> `class_weighting = balanced`
-> `batch_size = 32`
-> `num_epochs = 32`
-> `hidden_size = 128`
-> `num_hidden_layers = 8`
-> `num_attention_heads = 8`
-> `intermediate_size = 512`
-> `learning_rate = 3e-4`
-> `pooling = cls`
-> `device = mps`

O melhor checkpoint desta variante foi:

-> `best_epoch = 26`
-> `best_validation_macro_f1 = 0.8705`
-> `validation_accuracy = 0.8737`
-> `validation_loss = 0.4493`

Os gráficos desta variante foram guardados em:

-> `transformer/figures/pairwise_full_weighted_expanded_ffn512_plain_comparison_head_training_curves.png`
-> `transformer/figures/pairwise_full_weighted_expanded_ffn512_plain_comparison_head_validation_dashboard.png`

Comparação direta com a baseline anterior `plain + no comparison head`:

-> `0.8705 - 0.8694 = +0.0011` em `macro_f1`

Apesar do ganho ser pequeno, esta variante passou a ser a melhor configuração observada até agora.

## Resultado da Variante `Contextual + No Comparison Head`

Esta variante usa:

-> input contextualizado com `incident_type`, `detected_damages` e `vehicle`
-> classificador final sem comparison head explícito
-> dataset: `transformer/data/pairwise_dataset_full_contextual.jsonl`

O melhor checkpoint desta variante foi:

-> `best_epoch = 25`
-> `best_validation_macro_f1 = 0.8658`
-> `validation_accuracy = 0.8663`
-> `validation_loss = 0.3934`

Os gráficos desta variante foram guardados em:

-> `transformer/figures/pairwise_full_weighted_expanded_ffn512_context_no_comparison_head_training_curves.png`
-> `transformer/figures/pairwise_full_weighted_expanded_ffn512_context_no_comparison_head_validation_dashboard.png`

Comparações úteis:

-> contra `plain + no comparison head`: `0.8658 - 0.8694 = -0.0036`
-> contra `contextual + comparison head`: `0.8658 - 0.8580 = +0.0078`

Isto mostra que:

-> o comparison head foi útil quando o input permaneceu `plain`
-> a contextualização explícita do texto não trouxe ganho líquido
-> combinar contexto explícito com comparison head, na forma atual, foi a pior das quatro variantes comparadas

## Leitura Metodológica da Ablação

O principal resultado desta ablação é relativamente claro:

-> o ganho marginal veio do comparison head
-> o input contextualizado não melhorou a baseline
-> o melhor compromisso observado até agora é manter o texto mais simples e deixar a comparação ser aprendida pelo head final

Em termos de interpretação:

-> o comparison head parece ajudar a modelar relação entre os dois segmentos sem empurrar demasiados exemplos para `neutral`
-> a contextualização explícita com `incident_type` e `detected_damages` alterou a geometria de decisão, mas não melhorou a métrica final
-> a hipótese mais plausível é que o contexto textual extra introduziu ruído ou redundância para o encoder, em vez de fornecer um sinal suficientemente limpo

Neste ponto, a nova baseline principal do projeto passa a ser:

-> `plain + balanced weighting + 8 layers + intermediate_size 512 + comparison head`

## Extensão Multitask para Tipo de Inconsistência

Depois de estabilizar a melhor baseline pairwise single-task, foi implementada uma extensão multitask com um objetivo adicional:

-> além de prever `supports`, `neutral` ou `contradicts`, o modelo passa também a prever um `inconsistency_type`

A taxonomia atual desta segunda tarefa é:

-> `none`
-> `damage_mismatch`
-> `dynamics_mismatch`
-> `phantom_vehicle`
-> `scripted_narrative`

Esta segunda cabeça não é treinada com anotações manuais gold. Ela é derivada do corpus sintético e dos `fraud_indicators`, pelo que continua a ser weakly supervised.

## Resultado do Modelo Multitask

Foi executado um treino multitask mantendo a melhor configuração estrutural observada até aqui:

-> dataset: `transformer/data/pairwise_dataset_full.jsonl`
-> total de exemplos: `4752`
-> split estratificado: `3802` treino e `950` validação
-> `class_weighting = balanced`
-> `batch_size = 32`
-> `num_epochs = 32`
-> `hidden_size = 128`
-> `num_hidden_layers = 8`
-> `num_attention_heads = 8`
-> `intermediate_size = 512`
-> `learning_rate = 3e-4`
-> `pooling = cls`
-> `use_pairwise_comparison_head = True`
-> `use_inconsistency_head = True`
-> `inconsistency_loss_weight = 0.5`

O melhor checkpoint deste run foi:

-> `best_epoch = 22`
-> `best_validation_macro_f1 = 0.8659`

A avaliação do checkpoint em validação mostrou:

-> tarefa principal de relação:
-> `accuracy = 0.8705`
-> `macro_f1 = 0.8659`
-> tarefa auxiliar de `inconsistency_type`:
-> `inconsistency_accuracy = 0.8989`
-> `inconsistency_macro_f1 = 0.6109`

É importante interpretar estes números com cuidado:

-> a `accuracy` da tarefa auxiliar é alta em parte porque a classe `none` domina fortemente o corpus
-> a métrica mais informativa aqui é `inconsistency_macro_f1 = 0.6109`

## Leitura do Resultado Multitask

Comparação com a melhor baseline single-task:

-> baseline principal single-task: `macro_f1 = 0.8705`
-> modelo multitask: `macro_f1 = 0.8659`
-> diferença absoluta: `-0.0046`

Ou seja:

-> o multitask não superou a melhor baseline de classificação relacional
-> mas conseguiu aprender uma tarefa auxiliar não trivial

Na cabeça auxiliar, os resultados por classe mostram um padrão misto:

-> `none`: `f1 = 0.9664`
-> `damage_mismatch`: `f1 = 0.6957`
-> `dynamics_mismatch`: `f1 = 0.5192`
-> `phantom_vehicle`: `f1 = 0.7733`
-> `scripted_narrative`: `f1 = 0.1000`

Isto sugere:

-> o modelo já consegue recuperar bem inconsistências mais ancoradas em evidência explícita, como `phantom_vehicle` e `damage_mismatch`
-> `dynamics_mismatch` continua consideravelmente mais difícil
-> `scripted_narrative` ainda está muito fraco, o que é compatível com o baixo número de exemplos dessa classe

## Teste Manual com `predict_pairwise.py`

Foi também criado um utilitário de inferência:

-> `transformer/predict_pairwise.py`

Esse script já consegue devolver:

-> relação prevista do par
-> tipo de inconsistência previsto
-> interpretação textual curta do tipo previsto

No entanto, um teste manual com um par claramente contraditório ao nível da dinâmica do acidente devolveu:

-> `Predicted relation: supports`
-> `Predicted inconsistency type: scripted_narrative`

Isto é relevante porque mostra que:

-> a cabeça auxiliar existe e produz saídas estruturadas
-> mas o sistema ainda não está suficientemente fiável para responder, caso a caso, “o que não bate” com robustez operacional

Em termos metodológicos, a leitura correta neste ponto é:

-> o multitask é uma linha promissora para explicabilidade
-> mas ainda não substitui a baseline principal single-task
-> a inferência caso a caso ainda precisa de melhor supervisão ou melhor taxonomia para ser academicamente convincente

## Como Isto Encaixa no Pipeline de Fraude

O pipeline conceptual neste momento é:

-> Primeiro, detetar suporte, neutralidade ou contradição entre duas statements
-> Depois, usar várias decisões locais para resumir o comportamento de um claim completo
-> Só numa etapa posterior fazer agregação ao nível de fraude do claim

Isto significa que a arquitetura atual é uma etapa intermédia importante:

-> ainda não decide fraude diretamente ao nível global do claim
-> mas já aprende relações locais entre narrativas, que são essenciais para detetar inconsistências

## Current Status

Implementado neste momento:

-> tokenizer BPE customizado
-> tokens especiais do domínio
-> normalização de velocidade antes da tokenização
-> limpeza e deduplicação do corpus de tokenização
-> encoder pairwise em PyTorch
-> dataset encoding para pares anotados
-> builder weakly supervised para gerar dataset pairwise a partir dos claims existentes
-> integração do novo dataset `mixed_diverse` no corpus de treino
-> tokenizer retreinado sobre `3567` claims limpos
-> dataset pairwise balanceado com `2610` pares
-> dataset pairwise completo com `4752` pares
-> script de treino supervisionado baseline
-> suporte a treino no dataset completo com `class weighting`
-> builder pairwise com modos `plain` e `contextual`
-> baseline principal fixada em `plain + balanced weighting + 8 layers + intermediate_size 512 + comparison head`
-> variante contextualizada com `incident_type + detected_damages` no input pairwise implementada
-> pairwise comparison head explícita implementada
-> cabeça multitask para `inconsistency_type` implementada
-> utilitário de inferência `predict_pairwise.py` implementado
-> checkpointing do melhor modelo por `validation macro_f1`
-> script de avaliação de checkpoints
-> utilitário para gerar gráficos de treino
-> resultado experimental do corpus expandido documentado
-> pequena sweep arquitetural documentada
-> sanity test do modelo

Ainda em falta:

-> dataset pairwise anotado real
-> split de teste fixo para avaliação final
-> protocolo experimental comparativo mais completo
-> agregação ao nível do claim

## Limitações e Assunções Atuais

As principais assunções atuais são:

-> o tokenizer já existe e é carregado a partir de `transformer/tokenizers/claims_bpe`
-> o dataset pairwise atualmente disponível é weakly supervised e não uma anotação manual gold
-> o treino baseline usa `pairwise_dataset.jsonl` por defeito, ou seja, a versão balanceada
-> a variante `pairwise_dataset_full.jsonl` deve, em princípio, ser usada com `class weighting`
-> o modelo atual é uma baseline académica e não uma arquitetura otimizada para produção
-> o treino atual ainda não usa scheduler nem early stopping
-> mais profundidade não implica automaticamente melhor generalização neste corpus
-> mais volume de dados sintéticos também não implica automaticamente melhor `macro_f1`
-> mais contexto explícito no input pairwise também não implica automaticamente melhor `macro_f1`
-> o ganho atual do comparison head existe, mas ainda é pequeno e precisa de validação com múltiplas seeds
-> a tarefa auxiliar de `inconsistency_type` continua weakly supervised e depende da qualidade dos `fraud_indicators`

Também é importante ser explícito sobre o que ainda não existe no código:

-> não há claim-level aggregation model
-> não há pipeline final de fraude ponta a ponta
-> o builder atual ainda usa heurísticas conservadoras e não explora todas as combinações possíveis entre statements
-> a avaliação atual reutiliza split train/validation e ainda não corresponde a um benchmark final com holdout externo
-> ainda não existe comparação experimental completa entre `balanced sem weighting`, `full i256` e `full ffn512` com múltiplas seeds
-> ainda não existe estudo mais amplo de sensibilidade a seed, batch size e depth
-> ainda não existe estudo mais fino sobre como injetar contexto estrutural do claim sem degradar a fronteira `supports` vs `neutral`
-> o corpus expandido continua a ser weakly supervised e pode introduzir ruído semântico adicional
-> o `predict_pairwise.py` já devolve um tipo de inconsistência, mas esse output ainda não é suficientemente fiável para ser tratado como explicação definitiva do erro

## Next Step

O próximo passo mais correto, neste momento, já não é aumentar arquitetura. Também não é insistir imediatamente em mais heads auxiliares.

A prioridade passa a ser melhorar a supervisão da explicabilidade:

-> construir um subconjunto pairwise anotado manualmente com `inconsistency_type`
-> começar pelas classes que o multitask já mostrou conseguir aprender parcialmente:
-> `damage_mismatch`
-> `dynamics_mismatch`
-> `phantom_vehicle`

Só depois disso faz sentido:

-> recalibrar a taxonomia auxiliar
-> retreinar o multitask com labels mais fiáveis
-> reavaliar se o `predict_pairwise.py` realmente consegue dizer, com consistência, o que não bate entre duas histórias

Em resumo:

-> a baseline principal continua a ser o modelo single-task `plain + comparison head`
-> a linha de investigação mais promissora agora é transformar a cabeça auxiliar de `inconsistency_type` numa tarefa melhor supervisionada
