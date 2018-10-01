# Arabic-grammar-correction-Classifier
Arabic grammar correction Classifier
To Build an Arabic Grammar Analyzer several methodologies that can be followed to achieve the desired goal. 
We will present some of them:
• Conducting the analysis of the sentence input, both Morphology and syntactic
 using tools such as Alkalil(for morphological analysis) and Stanford (for syntactic analysis).
Thus, each word has its own analysis (what is the prefix, suffix), as well as its syntactic analysis (what is the grammatical relationship between the words that make up the sentence)
After obtaining the above information, Arabic grammar can be applied to the sentence, to find inconsistencies between the grammatical relationship and the Arabic grammatical rules on the relationship (such as the correlation between the verb and the subject).
This can be done by building a classifier for each grammatical relationship, so that we train it By giving income is a word feature of the grammatical relationship, and the classifier output is the characteristics of the word to which this relationship applies. 
We followed this method, but because there is no grammatical parser for the Arabic language with good accuracy, the grammatical correction was considered unobserved

we use this features for each word:
- Pos(stanford part of speech tag)
- Dependency (Stanford dependency parser)
- Gender(Named Entities)
- Number(using suffix & prefix)
- Prefix
- Suffix
