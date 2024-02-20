import random
import unicodedata
from cleantext import clean
import unicodedata
from annotation_processor import AnnotationProcessor
from utils.wiki_page import WikiPage
from database.feverous_db import FeverousDB
import logging
import re
import copy
def content_regularization(text):
    pattern = re.compile(r'\[{2}[^\|]*\|{1}[^\|]*\]{2}')  # find all entity-mention pairs in the form of [[entity|mention]]
    res = pattern.findall(text)
    for s in res:
        mention = s.strip().split('|')[1][:-2]
        text = text.replace(s, mention)
    return text

wiki_path = 'feverous_wikiv1.db'
DB = FeverousDB(wiki_path)
dev_retrieved_path = 'dual_channel_feverous/data/dev.combined.not_precomputed.p5.s5.t3.cells.jsonl'
test_retrieved_path = 'dual_channel_feverous/data/test.combined.not_precomputed.p5.s5.t3.cells.jsonl'
train_retrieved_path = 'dual_channel_feverous/data/train.combined.not_precomputed.p5.s5.t3.cells.jsonl'

old_dev_retrieved_path = '/data/xuweizhi/FEVEROUS-main/data/dev.combined.not_precomputed.p5.s5.t3.cells.jsonl'
old_train_retrieved_path = '/data/xuweizhi/FEVEROUS-main/data/train.combined.not_precomputed.p5.s5.t3.cells.jsonl'

# smalltrainpath = 'traintest.jsonl'


def get_wikipage_by_id(id):
    page = id.split('_')[0]
    # page = clean_title(page) legacy function used for old train/dev set. Not needed with current data version.
    page = unicodedata.normalize('NFD', page).strip()
    lines = DB.get_doc_json(page)
    if lines == None:
        print('Could not find page in database. Please ensure that the title is formatted correctly. If you using an old version (earlier than 04. June 2021, dowload the train and dev splits again and replace them in the directory accordingly.')
    pa = WikiPage(page, lines)
    return pa

def prepare_input(annotation, gold, indicator=False):
    sequence = [annotation.claim]
    evidence_type = ['claim']
    gold_indicator = []
    if gold:
        evidence_by_page = get_evidence_by_page(annotation.flat_evidence)
    else:
        evidence_by_page = get_evidence_by_page(annotation.predicted_evidence)
    
    if indicator and (not gold):
        gold_evidence = []
        for k in get_evidence_by_page(annotation.flat_evidence):
            gold_evidence.extend(k)
        gold_evidence = set(gold_evidence)
    
    for ele in evidence_by_page:
        for evid in ele:
            wiki_page = get_wikipage_by_id(evid)
            if '_sentence_' in evid:
                sequence.append('. '.join(
                    [str(context) for context in wiki_page.get_context(evid)[1:]]) + ' ' + get_evidence_text_by_id(evid,
                                                                                                                   wiki_page))
                evidence_type.append('sentence')
                if indicator and (not gold):
                    gold_indicator.append(1 if evid in gold_evidence else 0)
        tables = get_evidence_by_table(ele)
      
        for table in tables:
            content , content_order = linearize_cell_evidence_2d(table)
            assert(len(content)==len(content_order))
            sequence += content
            evidence_type.extend(['table' if 'cell' in ei else 'sentence' for ei in content_order])
            if indicator and (not gold):
                gold_indicator.extend([1 if ee in gold_evidence else 0 for ee in content_order])
    if indicator and gold:
        gold_indicator = [1]*(len(sequence)-1)
    assert(len(sequence)==len(evidence_type))
    if indicator:
        assert(len(sequence)==len(gold_indicator)+1)
    return [content_regularization(s) for s in sequence],evidence_type,gold_indicator

def linearize_cell_evidence_2d(table):
    context = []
    content_order = []

    # caption_id = [ele for ele in table if '_caption_' in ele]
    wiki_title = table[0].split("_")[0]
    # context.append(wiki_title)
    # if len(caption_id) > 0:
    #     wiki_page = get_wikipage_by_id(caption_id[0])
    #     context.append(get_evidence_text_by_id(caption_id[0], wiki_page))
    #     content_order.append('Added')
    cell_headers, cell_headers_type, table_type,cell_headers_content_id = group_evidence_by_header_2d(table)
    for key, values in cell_headers.items():
        if key == "caption":
            context.append(list(values)[0])
            content_order.append(cell_headers_content_id[key][0])
            continue
        if key == "item":
            lin = []
            for i, value in enumerate(values):
                lin.append(value.strip())
            context.extend(lin)
            content_order.extend(cell_headers_content_id[key])
            continue

        wiki_page = get_wikipage_by_id(key)
        lin = []
        # print(key, key_text, values)
        for i, value in enumerate(values):
            row_header, col_header, val = value

            if table_type == "infobox":
                lin.append( col_header.replace("[H] ", '').strip() + " : " + row_header.replace("[H] ",
                                                                                           '').strip() + " of " + wiki_title + " is " + val.strip())
                # lin += row_header.replace("[H] ", '').strip() + " of " + wiki_title + " is " + val.strip()
            elif table_type == "general":
                lin.append( col_header.replace("[H] ", '').strip() + " for " + row_header.replace("[H] ",
                                                                                             '').strip() + " is " + val.strip())
                lin[-1] = lin[-1].replace("for  ", '')
                # lin += col_header.replace("[H] ", '').strip() + " for " + row_header.replace("[H] ", '').strip() + " of " + wiki_title + " is " + val.strip()
                # lin += key_text.split('[H] ')[1].strip() + ' is ' + value #+ ' : ' + cell_headers_type[key]
            else:
                print(table_type)
                # assert False
        context.extend(lin)
        content_order.extend(cell_headers_content_id[key])
    assert(len(context)==len(content_order))
    return context,content_order

def get_evidence_by_page(evidence):
    evidence_by_page = {}
    for i, ele in enumerate(evidence):
        page = ele.split("_")[0]
        # page = str(i)
        if page in evidence_by_page:
            evidence_by_page[page].append(ele)
        else:
            evidence_by_page[page] = [ele]
    evis = [list(values) for key, values in evidence_by_page.items()]
    random.shuffle(evis)
    return evis

def get_evidence_by_table(evidence):
    evidence_by_table = {}
    for ev in evidence:
        if '_cell_' in ev:
            table = ev.split("_cell_")[1].split('_')[0]
        elif '_caption_' in ev:
            table = ev.split("_caption_")[1].split('_')[0]
        else:
            continue
        if table in evidence_by_table:
            evidence_by_table[table].append(ev)
        else:
            evidence_by_table[table] = [ev]
    return [list(values) for key, values in evidence_by_table.items()]

def get_evidence_text_by_id(id, wikipage):
    id_org = id
    id = '_'.join(id.split('_')[1:])
    if id.startswith('cell_') or id.startswith('header_cell_'):
        content = wikipage.get_cell_content(id)
    elif id.startswith('item_'):
        content = wikipage.get_item_content(id)
    elif '_caption' in id:
        content = wikipage.get_caption_content(id)
    else:
        if id in wikipage.get_page_items(): #Filters annotations that are not in the most recent Wikidump (due to additionally removed pages)
            content = str(wikipage.get_page_items()[id])
        else:
            print('Evidence text: {} in {} not found.'.format(id, id_org))
            content = ''
    return content


def group_evidence_by_header_2d(table):
    def get_cell_name(ele, header):
        return ele.split('_')[0] + '_' + header.get_id().replace('hc_', 'header_cell_')

    table_type = ''
    cell_headers = {}
    cell_headers_content_id = {}
    for ele in table:
        wiki_page = get_wikipage_by_id(ele)
        if 'header_cell_' in ele:
            continue #Ignore evidence header cells for now, probably an exception anyways
        elif "_item_" in ele:
            # continue
            context = wiki_page.get_context('_'.join(ele.split('_')[1:]))
            caption_str = ''
            for ct in context:
                if 'title' in ct.name:
                    caption_str += f"Title : {ct.content} , "
                elif 'section' in ct.name:
                    caption_str += f"Section : {ct.content} , "
                else:
                    assert False, ct.name
            caption_str += "Item : " + get_evidence_text_by_id(ele, wiki_page)
            # caption_str = [None, None, caption_str]
            if "item" in cell_headers:
                cell_headers["item"].append(caption_str)
                cell_headers_content_id["item"].append(ele)
            else:
                cell_headers["item"] = [caption_str]
                cell_headers_content_id["item"] = [ele]

        elif "_caption_" in ele:
            # continue
            context = wiki_page.get_context('_'.join(ele.split('_')[1:]))
            caption_str = ''
            for ct in context:
                if 'title' in ct.name:
                    caption_str += f"Title : {ct.content} , "
                elif 'section' in ct.name:
                    caption_str += f"Section : {ct.content} , "
                else:
                    assert False, ct.name
            caption_str += "Caption : " + get_evidence_text_by_id(ele, wiki_page)
            # caption_str = [None, None, caption_str]
            cell_headers["caption"] = [caption_str]
            cell_headers_content_id["caption"] = [ele]
        else:
            table_type, headers = find_headers('_'.join(ele.split('_')[1:]), wiki_page)
            # if table_type is None:
            #     assert False, print(ele)
            #     continue
            row_headers, col_headers = headers
            # row_headers = [ele.split('_')[0] + '_' + row_header.get_id().replace('hc_', 'header_cell_') for row_header in row_headers]
            # col_headers = [ele.split('_')[0] + '_' + col_header.get_id().replace('hc_', 'header_cell_') for col_header in col_headers]
            row_header = row_headers[0]
            col_header = col_headers[0]
            cell_header_ele = []
            # if table_type == "infobox":
            #     cell_header_ele.append(get_cell_name(ele, row_header))
            if "header_cell_" in col_header.get_id():
                cell_header_ele.append(get_cell_name(ele, col_header))
            elif "header_cell_" in row_header.get_id():
                cell_header_ele.append(get_cell_name(ele, row_header))
            else:
                cell_header_ele.append(get_cell_name(ele, col_header))

            # cell_header_ele = [ele.split('_')[0] + '_' +  el.get_id().replace('hc_', 'header_cell_') for el in wiki_page.get_context('_'.join(ele.split('_')[1:])) if "header_cell_" in el.get_id()]
            for head in cell_header_ele:
                cell_item = (get_evidence_text_by_id(get_cell_name(ele, row_header), wiki_page)
                             , get_evidence_text_by_id(get_cell_name(ele, col_header), wiki_page)
                             , get_evidence_text_by_id(ele, wiki_page))
                if head in cell_headers:
                    cell_headers[head].append(cell_item)
                    cell_headers_content_id[head].append(ele)
                else:
                    cell_headers[head] = [cell_item]
                    cell_headers_content_id[head] = [ele]
    cell_headers_type = {}
    for key in cell_headers.keys():
        assert(len(cell_headers[key])==len(cell_headers_content_id[key]))
    
    for key in cell_headers.keys():
        cell_headers_content_id[key],cell_headers[key]=remove_dup(cell_headers_content_id[key],cell_headers[key])
    # for ele, value in cell_headers.items():
    #     cell_headers[ele] = set(value)

    # for key,item in cell_headers.items():
    #     cell_headers_type[key] = calculate_header_type(item)

    return cell_headers, cell_headers_type, table_type,cell_headers_content_id

def remove_dup(ref,ob):
        ind = [0]*len(ref)
        showed = []
        for count,i in enumerate(ref):
            if i not in showed:
                showed.append(i)
            else:
                ind[count] = 1
        ref_,ob_ = [],[]
        for count in range(len(ref)):
            if ind[count]==1:
                continue
            else:
                ref_.append(ref[count])
                ob_.append(ob[count])
        return ref_,ob_
def find_headers(cell, page):
    table = None
    for ele in page.get_tables():
        if ele.name.split('_')[-1] == cell.split('_')[1]:
            table = ele
            break
    if table is None:
        logging.warning("Table not found in context, {}".format(cell))
        # return None, None

    cell_row = table.all_cells[cell].row_num
    cell_col = table.all_cells[cell].col_num
    headers_row = [cell for i, cell in enumerate(table.rows[cell_row].row) if cell_col > i]
    headers_row.reverse()
    context_row = set([])
    encountered_header = False
    for ele in headers_row:
        if ele.is_header:
            context_row.add(ele)
            encountered_header = True
        elif encountered_header:
            break

    if not context_row:
        try:
            for idx in range(cell_row):
                if table.rows[cell_row].row[idx].content:
                    context_row.add(table.rows[cell_row].row[idx])
                    break
        except IndexError:
            pass
    if not context_row:
        context_row.add(table.rows[cell_row].row[0])

    headers_column = [row.row[cell_col] for row in table.rows if cell_row > row.row_num]
    headers_column.reverse()
    context_column = set([])
    encountered_header = False
    for ele in headers_column:
        if ele.is_header:
            context_column.add(ele)
            encountered_header = True
        elif encountered_header:
            break
    if not context_column:
        try:
            for idx in range(cell_col):
                if table.rows[idx].row[cell_col].content:
                    context_column.add(table.rows[idx].row[cell_col])
                    break
        except IndexError:
            pass
    if not context_column:
        context_column.add(table.rows[0].row[cell_col])

    return table.type, [list(context_row), list(context_column)]



if __name__ == '__main__':
    devannotationlist = AnnotationProcessor(dev_retrieved_path)
    for i,annotation in enumerate(devannotationlist):
        if i==1:
            break
        print(annotation.verdict)

