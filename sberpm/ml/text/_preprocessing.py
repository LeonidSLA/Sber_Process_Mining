from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams
from ..._holder import DataHolder
from typing import Union, Any, Dict, Set, List, Iterable


class TextPreprocessing:
    """
    Class that preprocesses text, i.e. tokenizes into alphabetic tokens,
    transforms to normal forms, removes stop words, and creates n-grams.

    Parameters
    -----------
    language : {'rus', 'eng'}, default='rus'
        Text language, 'rus' for Russian and 'eng' for English.

    ngram : int, default=1
        Degree of the n-grams, 1 for unigrams, 2 for bigrams, 3 for trigrams, etc.

    Examples
    --------
    >>> from sberpm.ml.text import TextPreprocessing
    >>> preprocessing = TextPreprocessing()
    >>> preprocessed_text = preprocessing.transform(data_holder)
    """
    def __init__(self, language: str = 'rus',
                 ngram: int = 1) -> None:
        self._ngram = ngram
        self._morph_analyzer = MorphAnalyzer()
        self._language = language
        pattern = r'[а-яА-ЯЁё]+' if language == 'rus' else r'[A-Za-z]+'
        self._tokenizer = RegexpTokenizer(pattern)
        self._skip_words = self._init_skip_words()

    def _init_skip_words(self) -> Set[str]:
        """
        Returns a set of words to remove from the text.

        Returns
        -------
        skip_words : set
            Set of words to remove.
        """
        if self._language == 'rus':
            stop_words = stopwords.words('russian')
            stop_words.extend(['ваш', 'весь', 'всё', 'ещё', 'который', 'мочь', 'наш', 'оно', 'свой', 'это',
                               'кстати', 'вообще', 'тд', 'ой', 'туда', 'хотя', 'либо'])
            not_stop_words = ['без', 'более', 'всегда', 'иногда', 'лучше', 'много', 'можно', 'надо', 'нельзя',
                              'никогда', 'сейчас', 'хорошо', 'да', 'нет', 'не']
            for word in not_stop_words:
                stop_words.remove(word)
        else:
            stop_words = stopwords.words('english')
        return set(stop_words)

    def _get_normal_form(self, word: str,
                         all_words: Dict[str, str]) -> str:
        """
        Returns a common base form of the given word.

        Parameters
        ----------
        word : str
            Word to return a normal form for.

        all_words : dict
            Dictionary with words and their normal forms.

        Returns
        -------
        word : str
            Normal form of the word.
        """
        if word not in all_words:
            word_normal = self._morph_analyzer.parse(word)[0].normal_form
            all_words[word] = word_normal
        return all_words[word]

    def _preprocess_text(self, text: str,
                         all_words: Dict[str, str]) -> List[str]:
        """
        Returns a preprocessed text.

        Parameters
        ----------
        text : str
            Text to preprocess.

        all_words : dict
            Dictionary with words and their normal forms.

        Returns
        -------
        text : list
            List of tokens converted into n-grams.
        """
        tokens = self._tokenizer.tokenize(text.lower())
        tokens = [self._get_normal_form(word, all_words) for word in tokens]
        tokens = [t for t in tokens if t not in self._skip_words]
        return ['_'.join(n_gram) for n_gram in ngrams(tokens, n=self._ngram)]

    def transform(self, data: Union[DataHolder, str, Any]) -> List[Iterable]:
        """
        Preprocesses the given text by performing the following steps:
            - tokenization into alphabetic tokens;
            - lemmatization;
            - removing stop words;
            - generating n-grams.

        Parameters
        ----------
        data : sberpm.DataHolder or str or Any
            Text data to preprocess.

        Returns
        -------
        text : list
            List of preprocessed texts.
        """
        if type(data) is DataHolder:
            text_col = data.get_text()
        elif type(data) is str:
            text_col = [data]
        else:
            text_col = data
        all_words = {}
        preprocessed_text = [self._preprocess_text(text, all_words) for text in text_col]
        return preprocessed_text
