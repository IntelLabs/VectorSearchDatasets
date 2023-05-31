from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, AutoTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import BatchEncoding
import torch
import time
import os, shutil
import gzip
from datasets import load_dataset
# the files used/created by pickle are temporary and don't pose any security issue
import pickle #nosec
import random
import numpy as np
import numpy.typing as npt
import nltk
import sys
from typing import Optional, Union


def tokenize_texts(ctx_tokenizer: AutoTokenizer, texts: list, max_length: Optional[int] = None,
                   doc_stride: Optional[int] = None,
                   text_type: Optional[str] = "context", save_sentences: Optional[bool] = False, \
                   fname_sentences: Optional[str] = None) -> BatchEncoding:
    if text_type == "context":
        if max_length == None:
            max_length = 64
            print("Setting max_length to", max_length)
        if doc_stride == None:
            doc_stride = int(max_length / 2)
            print("Setting doc_stride to", doc_stride)

    start = time.time()
    if text_type == "context":
        encoded_inputs = ctx_tokenizer(texts, padding=True, truncation=True, max_length=max_length, \
                                       return_overflowing_tokens=True, \
                                       stride=doc_stride, return_tensors="pt")
    elif text_type == "query":
        encoded_inputs = ctx_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    end = time.time()
    delta_time = end - start
    print(f'Tokenization for {len(texts)}, contexts took {delta_time:.2f} s')

    n_seq = len(encoded_inputs['input_ids'])
    if save_sentences:
        # Code to generate sentences from tokens
        sentences = []
        for i in range(n_seq):
            if not (i % 100000):
                print('Processing sentence', i, 'of', n_seq)
            sentences += [' '.join(encoded_inputs.tokens(i))]

        with open(fname_sentences, 'wb') as f:
            pickle.dump(sentences, f)
        del sentences

    return encoded_inputs


def generate_embeddings(model: Union[DPRContextEncoder, DPRQuestionEncoder], encoded_input: BatchEncoding, dim: int,
                        batch_size: int, fname_mmap: str) -> np.memmap:
    n_seq = len(encoded_input['input_ids'])
    token_embeddings_out = np.memmap(fname_mmap, dtype='float32', \
                                     mode='w+', shape=(n_seq, dim))

    print('Doing inference for', n_seq, 'sequences.')

    model.eval()

    num_batches = int(np.ceil(float(n_seq) / batch_size))
    batch_print = 100
    start1 = torch.cuda.Event(enable_timing=True)
    end1 = torch.cuda.Event(enable_timing=True)
    start1.record()
    with torch.no_grad():
        for batch in range(num_batches):

            batch_init = batch * batch_size
            batch_end = np.min([batch_init + batch_size, n_seq])

            token_embeddings = model(encoded_input['input_ids'][batch_init:batch_end].to(device),
                                     encoded_input['attention_mask'][batch_init:batch_end].to(device))
            token_embeddings_out[batch_init:batch_end, :] = token_embeddings.pooler_output.cpu().numpy()

            del token_embeddings

            if not (batch % batch_print):
                print('Doing inference for batch', batch, 'of', num_batches)

    end1.record()
    torch.cuda.synchronize()
    print(f'Inference for {n_seq}, sequences took {(start1.elapsed_time(end1) / 1000):.2f} s')

    return token_embeddings_out


def extract_queries(texts: list, nRequired: Optional[int] = None, questionRequired: Optional[bool] = False) -> list:
    if nRequired is None:
        nRequired = len(texts)

    print("Extracting", nRequired, "queries from", len(texts), "texts.")

    total_questions = 0
    queries = []
    i = 0
    while len(queries) < nRequired and i < len(texts):
        sent_text = nltk.sent_tokenize(texts[i])
        nSentences = len(sent_text)

        # If the current paragraph has at least one question mark, find
        # the senteces that have them and append them to the list of queries.
        if texts[i].find('?') > 0:
            for j in range(nSentences):
                if sent_text[j].find('?') > 0:
                    queries.append(sent_text[j])
                    total_questions += 1
        else:  # Otherwise, just choose one sentence from the paragraph at random.
            if not questionRequired:
                chosen_sent = random.sample(list(range(nSentences)), 1)[0]
                queries.append(sent_text[chosen_sent])
        i += 1

        if not i % 1000:
            print(len(queries), "queries appended so far, with", total_questions, "questions,",
                  100 * total_questions / len(queries))

    return queries


def get_tokenized_seqs_total_for_contexts(contexts: list, ctx_tokenizer: AutoTokenizer, max_length: int,
                                          doc_stride: int, text_type: str) -> int:
    encoded_contexts = tokenize_texts(ctx_tokenizer, contexts, max_length, doc_stride=doc_stride, \
                                      text_type=text_type, save_sentences=False)
    return len(encoded_contexts['input_ids'])


def fvecs_write_from_mmap(fname: str, m: npt.ArrayLike):
    n, d = m.shape
    m1 = np.memmap(fname, dtype='int32', mode='w+', shape=(n, d + 1))
    m1[:, 0] = d
    m1[:, 1:] = m.view('int32')


if __name__ == "__main__":
    doc_stride = 32  # doc_stride: stride parameter for the tokenizer, it defines the overlap
    #             between extracted sequences
    max_length = 64  # max_length: max_length parameter for the tokenizer, it defines the maximum
    #             length of extracted sequences
    dim = 768  # dim: Dimensionality of the generated embeddings, it is defined by the model
    num_embd = 10000000  # num_embd: Number of embeddings to generate
    generate_queries = False  # generate_queries: Set to True to generate queries with the query encoder
    #                   (dpr-question_encoder-single-nq-base). Otherwise, the
    #                   base vectors are generated with the context encoder
    #                   (dpr-ctx_encoder-single-nq-base)
    questionRequired = True  # questionRequired: If set to True, sentences with question marks are
    #                   prioritized during query generation.
    get_total_embeddings_only = False  # get_total_embeddings_only: Used to get the number of embeddings that
    #                            can be generated from a file. The embeddings are
    #                            NOT generated if set to True.
    init_file = int(sys.argv[1])  # Use init_file and end_file parameters to determine the range of files to
    # extract text from. For example, init_file=0, end_file=10 will process
    end_file = int(sys.argv[2])  # input files c4-train.00000-of-01024.json.gz to
    # c4-train.00010-of-01024.json.gz from the C4/en dataset.

    np.random.seed(0)
    random.seed(0)

    cache_folder = f'/home/username/.cache/huggingface/datasets/files{init_file}_{end_file}'
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    fname_prefix_out = 'c4-en'
    if generate_queries:
        dataset_dir = '/home/username/research/datasets/c4/en/validation/'
        fname_prefix = 'c4-validation.'
        total_files = 8
    else:
        dataset_dir = '/home/username/research/datasets/c4/en/train/'
        fname_prefix = 'c4-train.'
        total_files = 1024

    if generate_queries:
        batch_size = 256
        save_sentences = True
        fname_mmap = f'{dataset_dir}/embeddings/{fname_prefix_out}_queries_{int(num_embd / 1000)}k_files{init_file}_{end_file}.mmap'
        fname_fvecs = f'{dataset_dir}/embeddings/{fname_prefix_out}_queries_{int(num_embd / 1000)}k_files{init_file}_{end_file}.fvecs'
    else:
        batch_size = 512
        save_sentences = False
        fname_mmap = f'{dataset_dir}/embeddings/{fname_prefix_out}_base_{int(num_embd / 1000000)}M_files{init_file}_{end_file}.mmap'
        fname_fvecs = f'{dataset_dir}/embeddings/{fname_prefix_out}_base_{int(num_embd / 1000000)}M_files{init_file}_{end_file}.fvecs'
        fname_tokens_total = f'{dataset_dir}/total_embeddings_list_files{init_file}_{end_file}.pkl'

    torch.set_grad_enabled(False)

    ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    ctx_tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    q_tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

    device = "cuda:" + str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
    print('Using', device)
    if generate_queries:
        q_encoder = q_encoder.to(device)
    else:
        ctx_encoder = ctx_encoder.to(device)

    if not get_total_embeddings_only:
        # If needed, create folders where mmaps and final fvecs files will be saved
        if not os.path.exists(f'{dataset_dir}/mmaps'):
            os.makedirs(f'{dataset_dir}/mmaps')
        if not os.path.exists(f'{dataset_dir}/embeddings'):
            os.makedirs(f'{dataset_dir}/embeddings')
        # Create mmap where all output embeddings will be saved
        fname_mmap_aux = f'{dataset_dir}/mmaps/partial_embs_files{init_file}_{end_file}.mmap'
        if os.path.isfile(fname_mmap):
            print(f'File {fname_mmap} already exists. Sure you want to overwrite it?')
            sys.exit()
        embeddings = np.memmap(fname_mmap, dtype='float32', mode='w+', shape=(num_embd, dim))

    file_id = init_file
    curr_total_emb = 0
    init_emb = 0
    total_embeds_per_file = []
    total_tokens = 0
    while file_id < end_file and curr_total_emb < num_embd:

        dataset_fname = f'{dataset_dir}{fname_prefix}{file_id:05d}-of-{total_files:05d}'
        fname_sentences_query = f'{dataset_dir}sentences/{fname_prefix}{file_id:05d}-of-{total_files:05d}_ml{max_length}_ds{doc_stride}_query_files{init_file}_{end_file}.pkl'

        print('Processing file' + dataset_fname)

        # Unzip file
        with gzip.open(dataset_fname + '.json.gz', 'rb') as f_in:
            with open(dataset_fname + '.json', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Load json from decompressed file
        dataset = load_dataset('json', data_files=dataset_fname + '.json', cache_dir=cache_folder)

        # Remove unnecesary json to save storage
        # os.remove(dataset_fname + '.json')

        if get_total_embeddings_only:
            contexts = dataset['train']['text']
            print('Processing', len(dataset['train']['text']), 'texts.')

            text_type = "context"
            total_tokens_text = get_tokenized_seqs_total_for_contexts(contexts, ctx_tokenizer, max_length, \
                                                                      doc_stride, text_type)
            total_embeds_per_file.append((total_tokens_text, dataset_fname))
            total_tokens += total_tokens_text

            print('There are', total_tokens_text, 'embeddings for file', dataset_fname)
        else:
            if generate_queries:
                text_for_queries = dataset['train']['text']
                queries = extract_queries(text_for_queries, questionRequired=questionRequired)

                print('Encoding', len(queries), 'queries.')
                text_type = "query"
                encoded_queries = tokenize_texts(q_tokenizer, queries, text_type=text_type, \
                                                 save_sentences=save_sentences, fname_sentences=fname_sentences_query)
                print('Generating embeddings for queries.')
                embeddings_batch = generate_embeddings(q_encoder, encoded_queries, dim, batch_size, fname_mmap_aux)
                del encoded_queries

            else:
                contexts = dataset['train']['text']
                print('Processing', len(dataset['train']['text']), 'texts.')

                print('Encoding', len(contexts), 'contexts.')
                text_type = "context"
                encoded_contexts = tokenize_texts(ctx_tokenizer, contexts, max_length, doc_stride=doc_stride, \
                                                  text_type=text_type, save_sentences=save_sentences)
                print('Generating embeddings for contexts.')
                embeddings_batch = generate_embeddings(ctx_encoder, encoded_contexts, dim, batch_size, fname_mmap_aux)
                del encoded_contexts

            curr_total_emb += embeddings_batch.shape[0]
            print(curr_total_emb, 'embeddings generated so far,', embeddings_batch.shape[0], 'from file', dataset_fname)

            if curr_total_emb < num_embd:
                # keep all generated embeddings
                end_emb = init_emb + embeddings_batch.shape[0]
                embeddings[init_emb:end_emb] = embeddings_batch
            else:
                # complete the last embeddings and finish
                embeddings[init_emb:] = embeddings_batch[:(num_embd - init_emb)]

            init_emb += embeddings_batch.shape[0]

        print('Cleaning huggingface cache...')
        cache_folder += '/json/'
        for filename in os.listdir(cache_folder):
            file_path = os.path.join(cache_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path, ignore_errors=True)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

        file_id += 1

    # Save mmap file to fvecs format, keeping only the valid embeddings
    print('Saving embeddings to .fvecs format.')
    fvecs_write_from_mmap(fname_fvecs, embeddings[:curr_total_emb])

    print('Deleting auxiliary mmaps.')
    os.remove(fname_mmap)
    os.remove(fname_mmap_aux)

    if get_total_embeddings_only:
        print('There are a total of', total_tokens, 'to be extrated from all files.')
        with open(fname_tokens_total, 'wb') as f:
            pickle.dump(total_embeds_per_file, f)
