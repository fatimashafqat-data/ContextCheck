# Error analysis

Six illustrative recorded errors; explanations below are hypotheses.

## Source 7280: Hate speech false negative
Reference: Hate speech. Prediction: Offensive language.

```text
usertoken my bitch dont want your ugly ass gtfo ol peyton manning lookin ass
```

The tweet is a short personal insult with gendered profanity and an appearance comparison. The visible text does not make an identity-based attack unambiguous, despite the hate-speech reference label. This may reflect annotation ambiguity or context unavailable in the dataset.

## Source 6124: Hate speech false positive
Reference: Offensive language. Prediction: Hate speech.

```text
usertoken usertoken what do you mean nice try? i'm right nigga. you a faggot. if you wanna argue bout anything else continue
```

The exchange contains racial and homophobic slurs in a direct personal argument. The boundary between an insult and identity-based hate is ambiguous here; the reference annotation is offensive language. A model may associate the slurs strongly with hate speech without resolving their conversational use.

## Source 9683: Hate speech false positive
Reference: Offensive language. Prediction: Hate speech.

```text
hate that bitch rt usertoken ray donovan's wife is pissing me off
```

The post combines “hate” with a gendered insult in a complaint about a television character. Those words may resemble identity-based abuse in training, although the reference label is offensive language. The word “hate” alone does not establish a hate-speech attack.

## Source 10121: Hate speech false negative
Reference: Hate speech. Prediction: Offensive language.

```text
i can't fuck a scary hoe.
```

This very short sentence contains gendered derogatory language but little context. The classifier assigns offensive language while the reference label is hate speech; the distinction depends on how the gendered term is interpreted, not on a clear explicit group attack.

## Source 3556: Hate speech false positive
Reference: Offensive language. Prediction: Hate speech.

```text
usertoken u can just ask him to suck his dick not that hard white trash
```

Sexual profanity and the phrase “white trash” appear in a personal exchange. The model may associate the identity-linked phrase with hate speech; annotators labeled this example offensive language. Missing conversation context makes the intended target uncertain.

## Source 3664: Hate speech false negative
Reference: Hate speech. Prediction: Offensive language.

```text
usertoken and how am i assume shit ,hmm cuz u don't know what fuck u talking about crackers these days.
```

The phrase “crackers these days” generalizes an insult to a racial group, while the rest reads as an individual argument. A word-based model may give more weight to the surrounding profanity and miss the group-directed meaning reflected in the reference label.