"Extractive summarization" oзначава да намерим важни части в текста, 
и да ги съберем като резюме на текста дословно. "abstractive summarization" 
пресъздава важната част на текста по нов начин след интерпретация 
и изследването му чрез NLP техники. Документа е насочен към 
"Extractive summarization" и ще приложим две техники за резюмиране
Latent Semantic Analysis и Text Rank. Първо ще разгледаме стандарните 
стъпки при "Extractive summarization"

# Dataset 

Използваме newsroom dataset, при който са събирани статии години наред 
заедно като имаме налични и резюметата на статиите. Използваме първите 
1000 елемента oт тестовия dataset.

[Newsroom](https://github.com/lil-lab/newsroom)

# Стъпка 1

Създаване на междинна репрезентация на входния текст. 
Има два подхода: topic representation и indicator representation.

При topic representation намираме важните теми чрез SVD, 
докато при indicator representation oписваме изречения спрямо 
различни фактори: при TextRank е cos оценката между две изречения.

# Стъпка 2
Оценяме изречения спрямо избрания подход. Всяко изречение получава оценка. 
При topic representation оценката ни дава колко изречението обяснява 
темите в текста. При indicator representation оценката е просто 
агрегацията на избраните от нас фактори (ние ще имаме един)

# Стъпка 3
Избираме най-важните k изречения спрямо оценките, които формират нашето резюме

# Подходи

# Latent Semantic Analysis (LSA)

LSA e unsupervised метод. Първата стъпка е да създадем term-sentence matrix , 
където имаме всеки term от документа като ред и всяко изречение като стълб. 
Всяка клетка се попълва с теглото да думата в изречение с TF-IDF оцента 
После използваме - singular value decomposition (SVD), която трансформира
матрица в три матрици term-topic matrix, diagonal matrix, topic-sentence matrix. 
След това се използват различни техники за избиране на изречение. Тук е използвана 
техниката на "Gong and Liu".

# Теxt Rank

Вдъхновен от PageRank този метод представя изреченията като свързан граф, 
където, на база оценка колко близки са две изречение, създаваме връзки 
между два върха на графа. Близостта между две изречения се измерва с cos оценка.
Изречения, към които сочат повече други изречения най-вероятно са по-важни и 
трябва да участват в резюмето. Този метод не е специфичен за език и може да се 
и използва в различни езици.

# Eвристичен подход 

Вземаме първите k изречения на текста при максимална стойност 3. 
Този метод е само за ориентиране накъде се движим с основните методи.

# Oпределяне на k 

Спрямо наличния dataset забелязваме че преобладават по кратки резюметата 
(1 изречение) дори и за по големите текстове. Спираме се на по консервативен 
подход за определяне на k - log10(Брой изречения) при стойност на 
минимална стойност на k >= 1.

# Оценки
Между предоставеното резюме и кандидат резюмето
- cos oценка
- Rouge L оценка намираща най-дългата подредица

# Eксперименти

Използваме първите 1000 елемента от dataset newsroom. 
За всяко получено резюме, сравняваме с cos и rougeL оценките 
спрямо наличното ни. Изчисляваме средната оценка и медианата на получени резултати.  Използваме трите описани подхода.

## text-rank

### cos:
- avg: 0.25
- median: 0.2

### rouge
- avg: 0.16
- median: 0.08

## lsa

### cos
- avg: 0.26
- median: 0.15

### rouge
- avg: 0.15
- median 0.05

## heuristics:

### cos:
- avg: 0.33
- median: 0.22

### rouge:
- avg: 0.22
- median: 0.09

# Бъдеща разработка

- Фокусиране върху специфична област (пример медицински новини)
- Избиране на различни оценъчни техники за изречение и теми за LSA
- Използване на Rouge-1, Rouge-2 и Rouge-N като допълнителни оценки
- Разшираване на списъка с абревиатури

Връзки:

[Paper](https://www.researchgate.net/publication/220195824_Text_summarization_using_Latent_Semantic_Analysis)

[Medium Article](https://medium.com/sciforce/towards-automatic-text-summarization-extractive-methods-e8439cd54715)