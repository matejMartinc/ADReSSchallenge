from lemmagen3 import Lemmatizer

# first, list all supported languages
print(Lemmatizer.list_supported_languages())

# then, create few lemmatizer objects using ISO 639-1 language codes
# (English, Slovene and Russian)

lem_en = Lemmatizer('en')
lem_sl = Lemmatizer('sl')
lem_ru = Lemmatizer('ru')

# now lemmatize the word "cats" in all three languages
print(lem_en.lemmatize('cats'))
print(lem_sl.lemmatize('je'))
print(lem_ru.lemmatize('коты'))
