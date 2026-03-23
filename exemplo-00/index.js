import tf from '@tensorflow/tfjs-node';

async function trainModel(inputXs, outputYs) {
    const model = tf.sequential()

    //Primeira camada da rede :
    // entrada 7 posicoes (idade normalizada + 3 cores + 3 localizacoes)

    //80 neuronios, coloquei tudo isso pq tem pouca base de treino
    // quanto mais neuronios, mais complexidade a rede pode aprender
    //e consequentemente mais processamento ela vai usar

    //A ReLU age como um filtor 
    // Como se ela deixasse somente os dados interessantes seguirem viajem na rede
    // Se a informacao chegou nesse neuronio é positiva passa pra frente
    // se for zero ou negativa, pode jogar fora, não vai servir pra nada
    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }))


    // Saida 3 neuronios
    // um para cada categoria ["premium", "medium", "basic"]

    //activation softmax normaliza a saida em probabilidades
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }))

    //Complilando o model (Adaptative moment estimation )
    // é um treinador pessoal moderno para redes neurais 
    // ajusta os pesos de forma eficiente e inteligente
    //aprender com historico de erros e acertos 

    // loss: categoricalCrossentropy
    // Ele compara o que o modelo "acha" (os scores de cada categoria)
    // com a resposta certa
    // a categoria premium será sempre [1, 0, 0]

    // quanto mais distante da previsão do modelo da resposta correta
    // maior o erro (loss)
    // Exemplo classico: classificação de imagens, recomendação, categorização de
    // usuário
    // qualquer coisa em que a resposta certa é "apenas uma entre várias possíveis"
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    })

    // Treinamento do modelo
    // verbose: desabilita o log interno (e usa só callback)
    // epochs: quantidade de veses que vai rodar no dataset
    // shuffle: embaralha os dados, para evitar viés
    await model.fit(
        inputXs,
        outputYs,
        {
            verbose: 0,
            epochs: 100, 
            shuffle: true,
            callbacks: {
                // onEpochEnd: (epoch, log) => console.log(
                //     `Epoch ${epoch}: loss = ${log.loss}`
                // )
            }
        }
    )

    return model

}
// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

// quanto mais dado melhor!
// assim o algoritmo consegue entender melhor os padrões complexos
// dos dados
const model = trainModel(inputXs, outputYs)