from pprint import pprint
import wikipedia
import json
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS


def search(results=1):
    with open('./embedding_caches/custom_embeddings_freq_lc.json', 'r') as fp:
        voc = json.load(fp)

    pprint(voc)

    count = 0
    for word in sorted(voc.keys()):

        print(count, word)
        count += 1
        if word in ENGLISH_STOP_WORDS:
            continue
        response = wikipedia.search(word, results=results)
        print(word)
        print(response)
        voc[word] = response

    with open('./embedding_caches/wikipedia.json', 'w') as fp:
        json.dump(voc, fp)


if __name__ == '__main__':

    with open('./embedding_caches/wikipedia.json', 'r') as fp:
        responses = json.load(fp)

    concepts = {r[0] for r in responses.values() if isinstance(r, list) and len(r) > 0}

    for idx, concept in enumerate(sorted(concepts)[:]):
        print(idx, concept)
        try:
            page = wikipedia.page(concept)
        except wikipedia.exceptions.DisambiguationError as de1:
            try:
                page = wikipedia.page(de1.options[0])
            except wikipedia.exceptions.DisambiguationError as de2:
                try:
                    page = wikipedia.page(de2.options[1])
                except wikipedia.exceptions.DisambiguationError as de3:
                    print('wikipedia.exceptions.DisambiguationError ... giving up.')
                except wikipedia.exceptions.PageError as pe:
                    print('wikipedia.exceptions.PageError')
            except wikipedia.exceptions.PageError as pe:
                print('wikipedia.exceptions.PageError')
        except wikipedia.exceptions.PageError as pe:
            print('wikipedia.exceptions.PageError')

        print('title:', page.title)
        string = '{{__TITLE__}} '
        string += page.title + '\n'
        string += page.content
        string += '\n{{__END_DOC__}}\n'
        with open('./embedding_caches/wikipedia_corpus.txt', 'a') as fp:
            fp.write(string)
