import os
import json
import math
import random
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk import pos_tag, FreqDist


def read_dataset(in_dir, out_dir, num_sample=200):
    # read and sample dataset if not sample yet
    if not os.path.exists(out_dir):
        with open(in_dir, encoding='utf-8') as fin:
            all_reviews = fin.readlines()
            sampled_reviews = random.sample(all_reviews, num_sample)
            ds = [json.loads(item) for item in sampled_reviews]
            # write to file
            with open(out_dir, 'w', encoding='utf-8') as fout:
                for item in ds:
                    fout.write(json.dumps(item))
                    fout.write('\n')
    else:
        with open(out_dir, encoding='utf-8') as fin:
            ds = [json.loads(item) for item in fin.readlines()]
    return ds


def plot_sentence_seg(sentences1, sentences2):
    stats_dict = {}
    for reviews in sentences1:
        if len(reviews) not in stats_dict:
            stats_dict[len(reviews)] = [0, 0]
        stats_dict[len(reviews)][0] += 1

    for reviews in sentences2:
        if len(reviews) not in stats_dict:
            stats_dict[len(reviews)] = [0, 0]
        stats_dict[len(reviews)][1] += 1

    x_label = np.array(sorted(stats_dict.keys()))
    y_arr1 = [stats_dict[key][0] for key in x_label]
    y_arr2 = [stats_dict[key][1] for key in x_label]
    plt.figure(figsize=(10, 6))
    plt.title('Sentence Segmentation')
    plt.xlabel('the length of a review in number of sentences')
    plt.ylabel('the number of reviews of that length')
    x = np.arange(len(x_label))
    width = 0.4
    plt.bar(x - width/2, y_arr1, width=width, label='Health_and_Personal_Care')
    plt.bar(x + width/2, y_arr2, width=width, label='Video_Games')
    plt.xticks(x, x_label)
    plt.grid()
    plt.legend()
    plt.savefig('../plot/sentence_segmentation.png')
    # plt.show()


def plot_tokenization(tokens1, tokens2):
    stats_dict1 = {}
    stats_dict2 = {}
    for reviews in tokens1:
        if len(reviews) not in stats_dict1:
            stats_dict1[len(reviews)] = 0
        stats_dict1[len(reviews)] += 1

    for reviews in tokens2:
        if len(reviews) not in stats_dict2:
            stats_dict2[len(reviews)] = 0
        stats_dict2[len(reviews)] += 1

    x_arr1 = sorted(stats_dict1.keys())
    x_arr2 = sorted(stats_dict2.keys())
    y_arr1 = [stats_dict1[key] for key in x_arr1]
    y_arr2 = [stats_dict2[key] for key in x_arr2]

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    ax[0].set_title('Review Length (measured by the number of tokens) Distribution')
    ax[0].set_ylabel('the number of reviews of that length')
    ax[1].set_ylabel('the number of reviews of that length')
    ax[1].set_xlabel('the length of a review in number of tokens')

    ax[0].scatter(x_arr1, y_arr1, color='r', s=10, label='Health_and_Personal_Care')
    ax[1].scatter(x_arr2, y_arr2, color='b', s=10, label='Video_Games')
    ax[0].legend()
    ax[1].legend()
    ax[0].grid(True)
    ax[1].grid(True)
    fig.savefig('../plot/tokenization.png')
    # plt.show()


def plot_stemming(tokens_list, name_dataset):
    # count the number of unique tokens before stemming
    before_stem = FreqDist(itertools.chain.from_iterable(tokens_list))
    print(f'there are {before_stem.B()} unique tokens in {name_dataset} without stemming\n')
    # do stemming and count the number of unique tokens again
    stemmer_porter = PorterStemmer()
    stemmer_lancaster = LancasterStemmer()
    stemmer_snowball = SnowballStemmer('english')

    stemmed_p = [stemmer_porter.stem(token) for token in itertools.chain.from_iterable(tokens_list)]
    stemmed_l = [stemmer_lancaster.stem(token) for token in itertools.chain.from_iterable(tokens_list)]
    stemmed_s = [stemmer_snowball.stem(token) for token in itertools.chain.from_iterable(tokens_list)]

    stemmed_dict_p = FreqDist(stemmed_p)
    stemmed_dict_l = FreqDist(stemmed_l)
    stemmed_dict_s = FreqDist(stemmed_s)
    print(f'there are {stemmed_dict_p.B()} unique tokens in {name_dataset} with porter stemming\n')
    print(f'there are {stemmed_dict_l.B()} unique tokens in {name_dataset} with lancaster stemming\n')
    print(f'there are {stemmed_dict_s.B()} unique tokens in {name_dataset} with snowball stemming\n')

    fig = plt.figure(figsize=(10, 8))
    plt.title(f"{name_dataset} without Stemming")
    before_stem.plot(60, show=False)
    fig.savefig(f'../plot/{name_dataset}_wostem.png')

    fig = plt.figure(figsize=(10, 8))
    plt.title(f"{name_dataset} with Porter Stemming")
    stemmed_dict_p.plot(60, show=False)
    fig.savefig(f'../plot/{name_dataset}_pstem.png')

    fig = plt.figure(figsize=(10, 8))
    plt.title(f"{name_dataset} with Lancaster Stemming")
    stemmed_dict_l.plot(60, show=False)
    fig.savefig(f'../plot/{name_dataset}_lstem.png')

    fig = plt.figure(figsize=(10, 8))
    plt.title(f"{name_dataset} with Snowball Stemming")
    stemmed_dict_s.plot(60, show=False)
    fig.savefig(f'../plot/{name_dataset}_sstem.png')


def get_indicative_list(tokens_list1, tokens_list2):
    dict = {}
    characters = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!',
                  '*', '@', '#', '$', '%', '-', '...', '^', '{', '}']
    for tokens in tokens_list1:
        # remove characters and change to lower case
        words = [token.lower() for token in tokens if token not in characters]
        fd = FreqDist(words)
        for key in fd.keys():
            if key not in dict:
                dict[key] = [0, 0]
            dict[key][0] += 1

    for tokens in tokens_list2:
        # remove characters and change to lower case
        words = [token.lower() for token in tokens if token not in characters]
        fd = FreqDist(words)
        for key in fd.keys():
            if key not in dict:
                dict[key] = [0, 0]
            dict[key][1] += 1

    # add 1 for smoothing to avoid divided by zero
    indicative_ds1 = []
    indicative_ds2 = []
    for key in dict:
        indicative_ds1.append((key, (dict[key][0] + 1) * math.log((dict[key][0] + 1) / (dict[key][1] + 1))))
        indicative_ds2.append((key, (dict[key][1] + 1) * math.log((dict[key][1] + 1) / (dict[key][0] + 1))))

    return indicative_ds1, indicative_ds2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=42,
                        help='random seed')
    args = parser.parse_args()

    random.seed(args.random_seed)  # set random seed for reproducibility
    ds1_dir = '../data/Health_and_Personal_Care_5.json'
    ds2_dir = '../data/Video_Games_5.json'

    # read data and randomly sample 200 products
    ds1 = read_dataset(ds1_dir, os.path.join('../data', 'ds1_sampled.json'))
    ds2 = read_dataset(ds2_dir, os.path.join('../data', 'ds2_sampled.json'))
    reviews_ds1 = [item['reviewText'] for item in ds1]
    reviews_ds2 = [item['reviewText'] for item in ds2]

    # Sentence Segmentation
    sentences_ds1 = [sent_tokenize(item) for item in reviews_ds1]
    sentences_ds2 = [sent_tokenize(item) for item in reviews_ds2]
    plot_sentence_seg(sentences_ds1, sentences_ds2)

    # Tokenization and Stemming
    tokens_ds1 = [word_tokenize(item) for item in reviews_ds1]
    tokens_ds2 = [word_tokenize(item) for item in reviews_ds2]
    plot_tokenization(tokens_ds1, tokens_ds2)
    plot_stemming(tokens_ds1, 'Health_and_Personal_Care')
    plot_stemming(tokens_ds2, 'Video_Games')

    # POS Tagging
    selected_sentences = random.sample(list(itertools.chain.from_iterable(sentences_ds1)), 3) + \
                         random.sample(list(itertools.chain.from_iterable(sentences_ds2)), 2)
    with open('out.txt', 'a') as f:
        for sentence in selected_sentences:
            tokens = word_tokenize(sentence)
            pos_res = pos_tag(tokens)
            print(pos_res, file=f)

    # Indicative Words
    list1, list2 = get_indicative_list(tokens_ds1, tokens_ds2)
    sorted_list1 = sorted(list1, key=lambda item: item[1], reverse=True)
    sorted_list2 = sorted(list2, key=lambda item: item[1], reverse=True)
    print(sorted_list1[:10])
    print(sorted_list2[:10])


if __name__ == '__main__':
    main()
