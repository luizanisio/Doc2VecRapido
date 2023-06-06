# -*- coding: utf-8 -*-
# referencia: 06/06/2023
# https://github.com/luizanisio/Doc2VecRapido/blob/main/src/util_pandas.py

import pandas as pd
import os
from xlsxwriter.utility import xl_col_to_name

import regex as re 
from hashlib import sha1 as hl_sha1
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count
import numpy as np

class UtilTexto():
    ABREVIACOES = ['sra?s?', 'exm[ao]s?', 'ns?', 'nos?', 'doc', 'ac', 'publ', 'ex', 'lv', 'vlr?', 'vls?',
                'exmo\(a\)', 'ilmo\(a\)', 'av', 'of', 'min', 'livr?', 'co?ls?', 'univ', 'resp', 'cli', 'lb',
                'dra?s?', '[a-z]+r\(as?\)', 'ed', 'pa?g', 'cod', 'prof', 'op', 'plan', 'edf?', 'func', 'ch',
                'arts?', 'artigs?', 'artg', 'pars?', 'rel', 'tel', 'res', '[a-z]', 'vls?', 'gab', 'bel',
                'ilm[oa]', 'parc', 'proc', 'adv', 'vols?', 'cels?', 'pp', 'ex[ao]', 'eg', 'pl', 'ref',
                'reg', 'f[ilí]s?', 'inc', 'par', 'alin', 'fts', 'publ?', 'ex', 'v. em', 'v.rev',
                'des', 'des\(a\)', 'desemb']    
    ABREVIACOES_RGX = re.compile(r'(?:\b{})\.\s*$'.format(r'|\b'.join(ABREVIACOES)), re.IGNORECASE)
    PONTUACAO_FINAL = re.compile(r'([\.\?\!]\s+)')
    PONTUACAO_FINAL_LISTA = {'.','?','!'}
    RE_NUMEROPONTO = re.compile(r'(\d+)\.(?=\d)')
    RE_NAO_LETRAS = re.compile('[^\wá-ú]')
    # recebe uma lista de parágrafos quebrados (normalmente de um PDF)
    # e tenta identificar quais deveriam estar juntos de acordo com o final de
    # cada linha e o início da próxima
    # retorna uma lista de parágrafos
    @classmethod
    def unir_paragrafos_ocr(cls, texto):
        lista = texto if type(texto) is list else texto.split('\n')
        res = []
        def _final_pontuacao(_t):
            if len(_t.strip()) == 0:
                return False
            return _t.strip()[-1] in cls.PONTUACAO_FINAL_LISTA
        for i, linha in enumerate(lista):
            if i==0:
                res.append(linha)
            elif (not _final_pontuacao(lista[i-1])) or \
                (_final_pontuacao(lista[i-1]) and (cls.ABREVIACOES_RGX.search(lista[i-1]))):
                if len(res) ==0:
                   res =['']
                res[len(res)-1] = res[-1].strip() + ' '+ linha
            else:
                res.append(linha)
        return res

    @classmethod
    def hash_str(texto):
        _txt = '|'.join(texto) if type(texto) is list else str(texto)
        return hl_sha1(_txt.encode('utf-8')).hexdigest()    

class UtilPandas():
    
    @classmethod
    def split_dataframe(cls, df, num_batches):
        # Dividir o dataframe em partes iguais
        df_split = np.array_split(df, num_batches)
        df_split = [_ for _ in df_split if len(_) > 0]
        return df_split
    
    @classmethod
    def apply_parallel(cls, df, func, num_threads=None):
        num_threads = num_threads or cpu_count()

        df_split = cls.split_dataframe(df, num_threads)
       
        # Criar um pool de threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Aplicar a função em cada parte do dataframe usando o pool de threads
            results = executor.map(lambda _df:_df.apply(func, axis=1) , df_split)

        # Concatenar os resultados em um único dataframe
        return pd.concat(results)

    @classmethod
    def apply_process(cls, df, func, num_processes=None):
        num_processes = num_processes or cpu_count()

        df_split = cls.split_dataframe(df, num_processes)

        # Criar um pool de processos
        with Pool(processes=num_processes) as pool:
            # Aplicar a função em cada parte do dataframe usando o pool de processos
            df_func = [(_df, func) for _df in df_split]
            results = pool.map(cls.__func_apply__, df_func)

        # Concatenar os resultados em um único dataframe
        return pd.concat(results)
    
    @classmethod
    def __func_apply__(cls, df_func):
        df, func = df_func
        return df.apply(func, axis=1)
        
    ### EXEMPLO ##################################################
    @classmethod
    def __func_exemplo__(cls, item):
        item['C'] = item['A'] + item['B']
        print(f"item {item['A'].item()}")
        return item
    
    @classmethod
    def exemplo(cls, processos = True):
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        if processos:
            return cls.apply_process(df, cls.__func_exemplo__)
        return cls.apply_parallel(df, cls.__func_exemplo__)

#### Pandas Excel ################################################    
class UtilPandasExcel:
    FORMAT_HEADER = {'bold':  True, 'align': 'left', 'valign': 'top', 'text_wrap': True, 'bg_color': '#CCCCCC'}
    FORMAT_DEFAULT = {'bold':  False, 'align': 'left', 'valign': 'top', 'text_wrap': True}
    FORMAT_HIGHLIGHT = {'bold':  True, 'align': 'left', 'valign': 'top', 'text_wrap': True, 'bg_color': '#FFFD9A'}

    MAX_WIDTH = 100
    RE_MAIUSCULAS = re.compile('[A-Z]')

    def __str_size__(self,texto):
        sz = 0.0
        for lt in texto:
            sz = sz + (1.33 if self.RE_MAIUSCULAS.search(lt) else 1)
        return round(sz + 0.5)

    def __init__(self, nome_arquivo:str, columns_auto_width = True, header_formatting = True):
        if (nome_arquivo.lower()[-5:] != '.xlsx') and (nome_arquivo.lower()[-4:] != '.xls'):
            nome_arquivo = f'{nome_arquivo}.xlsx'
        self.nome_arquivo = nome_arquivo
        self.writer = pd.ExcelWriter(self.nome_arquivo, engine='xlsxwriter')
        self.columns_auto_width = columns_auto_width
        self.header_formatting = header_formatting
        self.WB_HEADER_FORMAT = self.writer.book.add_format(self.FORMAT_HEADER)
        self.WB_DEFAULT_FORMAT = self.writer.book.add_format(self.FORMAT_DEFAULT)
        self.WB_HIGHLIGHT_FORMAT = self.writer.book.add_format(self.FORMAT_HIGHLIGHT)
        # cores
        self.COR_CINZA_CLARO = self.writer.book.add_format({'bg_color': '#F2F2F2'})
        self.COR_CINZA = self.writer.book.add_format({'bg_color': '#A4A4A4'})
        self.COR_AZUL_CLARO = self.writer.book.add_format({'bg_color': '#E0F8F7'})
        self.COR_VERDE_CLARO = self.writer.book.add_format({'bg_color': '#CEF6EC'})
        self.COR_CREME = self.writer.book.add_format({'bg_color': '#F5FFFA'})
        self.COR_LARANJA = self.writer.book.add_format({'bg_color': '#FFDAB9'})

        self.FONTE_AZUL = self.writer.book.add_format({'font_color': '#0000FF'})
        self.FONTE_VERDE = self.writer.book.add_format({'font_color': '#088A08'})

    def nova_cor(self, fundo, fonte):
        res = {}
        if fundo:
            res['bg_color'] = f'{fundo}' 
        if fonte:
            res['font_color'] = f'{fonte}'
        return self.writer.book.add_format(res)


    def __get_cell_format__(self, row):
        if row % 2 == 0:
            return self.CELULAS_PARES
        else:
            return self.CELULAS_IMPARES

    def write_df(self,df,sheet_name : str, auto_width_colums_list = True, columns_titles = None):
        df.to_excel(self.writer, sheet_name=f'{sheet_name}', index = False)
        if not (columns_titles is None):
            worksheet = self.writer.sheets[f'{sheet_name}']
            for n, value in enumerate(columns_titles):
                worksheet.write(0, n, value)
        # formata os tamanhos das colunas
        self.__auto_width_colums__(df = df, sheet_name = sheet_name,columns_list = auto_width_colums_list, columns_titles=columns_titles)
        # formata o cabeçalho
        self.__format_header__(df = df, sheet_name=sheet_name)

    def write_dfs(self,dataframes: dict,auto_width_colums_list = True):
        for n,d in dataframes.items():
            self.write_df(df = d, sheet_name = n, auto_width_colums_list = auto_width_colums_list)

    # recebe o endereço da célula e o valor
    def write_cell(self, sheet_name : str, cell:str, value, is_header = False):
        worksheet = self.writer.sheets[f'{sheet_name}']
        fm = self.WB_HEADER_FORMAT if is_header else None 
        #worksheet.write(0, colx, value, self.WB_HEADER_FORMAT)        
        worksheet.write(f'{cell}', value, fm)

    # recebe a posição inicial e final e grava a lista na linha
    # exmeplos:
    # upd.write_cell(sheet_name='Resumo de Entidades', cell='E1', value = 'TREINO', is_header = True)
    # upd.write_cells(sheet_name='Resumo de Entidades', col=5,line=0, values = ['TIPO', 'INICIO','FIM','PAI','MEDIA'], is_header= True)
    # upd.write_cells(sheet_name='Resumo de Entidades', col=5,line=2, values = ['TIPO', 'INICIO','FIM','PAI','MEDIA'], is_header= False)
    def write_cells(self, sheet_name : str, col:int, line:int, values = [], is_header = False):
        worksheet = self.writer.sheets[f'{sheet_name}']
        fm = self.WB_HEADER_FORMAT if is_header else None 
        #worksheet.write(0, colx, value, self.WB_HEADER_FORMAT)        
        for n, value in enumerate(values):
            worksheet.write(line, col + n, value, fm)

    # recebe um json e grava uma tabela com cabeçalho
    # exmeplos:
    # upd.write_cells(sheet_name='Resumo de Entidades', col=5,line=4, values = [{'INICIO':1,'FIM':2},{'INICIO':3,'FIM':4}], is_header= False, col_order=['INICIO','FIM'])
    def write_table(self, sheet_name : str, col:int, line:int, values = [], is_header = False, col_order = None, columns_titles = None):
        worksheet = self.writer.sheets[f'{sheet_name}']
        fm = self.WB_HEADER_FORMAT if is_header else None 
        _col_order = list(col_order) if not col_order is None else None
        _col_title = _col_order if columns_titles is None else list(columns_titles)
        # se não receber a ordem, cria a lista com todas as colunas
        if _col_order is None:
            _col_order = []
            for value in values:
                for c in value.keys():
                    if not c in _col_order:
                        _col_order.append(c)
            _col_title = _col_order
        # grava a coluna de cabeçalhos
        self.write_cells(sheet_name=sheet_name, col=col, line=line, values = _col_title, is_header = True)
        _col_size = [self.__str_size__(c) + 2 for c in _col_title]
        # grava os dados
        for n, value in enumerate(values):
            for colx, k in enumerate(_col_order):
                vl = value.get(k)
                if not (vl is None):
                    worksheet.write(line + n + 1, col + colx, vl, fm)
                    _col_size[colx] = max(self.__str_size__(f'{vl}') , _col_size[colx])
        # ajusta a largura das células
        for n, s in enumerate(_col_size):
            _sz = min(self.MAX_WIDTH, s)
            worksheet.set_column(col + n, col + n, _sz)  # set column width   

    def __auto_width_colums__(self,df:pd.DataFrame, sheet_name : str, columns_list = [], columns_titles = None):
        if (columns_list == False ):
            return
        # inspirado em https://stackoverflow.com/questions/17326973/is-there-a-way-to-auto-adjust-excel-column-widths-with-pandas-excelwriter
        worksheet = self.writer.sheets[sheet_name]  # pull worksheet object
        for idx, col in enumerate(df):  # loop through all columns
            series = df[col]
            if (columns_list == True) or (len(columns_list) == 0) or (col in columns_list):
                max_len = max((
                    series.astype(str).map(self.__str_size__).max(),  # len of largest item
                    self.__str_size__(str(series.name))  # len of column name/header
                    )) + 2  # adding a little extra space
                if (columns_titles is not None) and (len(columns_titles)>=idx-1):
                    max_len = max((max_len,self.__str_size__(columns_titles[idx])))
                max_len = max_len if max_len <= self.MAX_WIDTH else self.MAX_WIDTH # verifica o maior tamanho de uma coluna
                worksheet.set_column(idx, idx, max_len)  # set column width        

    def __format_header__(self, df : pd.DataFrame , sheet_name : str):
        if not self.header_formatting:
            return
        # inspirado em https://stackoverflow.com/questions/39919548/xlsxwriter-trouble-formatting-pandas-dataframe-cells-using-xlsxwriter
        # Write the header manually
        worksheet = self.writer.sheets[f'{sheet_name}']
        for colx, value in enumerate(df.columns.values):
            worksheet.write(0, colx, value, self.WB_HEADER_FORMAT)        
        #worksheet.set(0,0,cell_format =self.WB_HEADER_FORMAT)

    def conditional_color(self, sheet_name, cells, min_value = 0, mid_value = 0.75, max_value = 1):
        # inspirado em https://xlsxwriter.readthedocs.io/working_with_conditional_formats.html
        worksheet = self.writer.sheets[f'{sheet_name}']
        #fm = FORMAT_CONDITIONAL_3_COLOR = {'type': '3_color_scale', min_value, mid_value, max_value}
        worksheet.conditional_format(f'{cells}', {'type': '3_color_scale', 'min_value':min_value, 'mid_value': mid_value, 'max_value':max_value})

    def highlight_bgcolor(self, sheet_name, cells,  min_value = 0, max_value = 1):
        # inspirado em https://xlsxwriter.readthedocs.io/working_with_conditional_formats.html
        worksheet = self.writer.sheets[f'{sheet_name}']
        worksheet.conditional_format(f'{cells}', {'type': 'cell', 'criteria': 'between','minimum':  min_value,'maximum':  max_value, 'format':   self.WB_HIGHLIGHT_FORMAT})

    # cria um range para a posição das colunas informadas
    def range_cols(self,first_col , last_col , first_row = None, last_row = None):
        _fr = '' if first_row is None else f'{first_row}'
        _lr = '' if last_row is None else f'{last_row}'
        return f'{xl_col_to_name(first_col)}{_fr}:{xl_col_to_name(last_col)}{_lr}'

    def conditional_value_color(self, sheet_name, cells, valor, cor = None):
        worksheet = self.writer.sheets[f'{sheet_name}']
        _cor = cor if cor else self.COR_CINZA
        worksheet.conditional_format(f'{cells}', {'type': 'formula', 'criteria': f'={valor}', 'format': _cor})

    RE_URL = re.compile(r'https?://')
    RE_FORMULA = re.compile(r'^(\s*\=)*')
    # substitui formulas e urls
    @staticmethod
    def clear_string(value):
        return UtilPandasExcel.RE_FORMULA.sub('', UtilPandasExcel.RE_URL.sub('((url))',value))

    def save(self):
        self.writer.save()
    
if __name__ == '__main__':
    print('Exemplo com processos:') 
    df = UtilPandas.exemplo(True)
    print(df)
    print('=========================================') 
    print('Exemplo com threads:')
    df = UtilPandas.exemplo(False)
    print(df)
    print('=========================================') 

    print('Exemplo com excel:')
    
    teste = [{"ano":2000, "quantidade":10, 'linha_grande' : "aka slkjsdlfjasldjf lasdjflsjdflaksjdlkf"},
             {"ano":2001, "quantidade":15, "linha_grande" : "dlsafalskjdf lflak sjdflasldfasldkfjsaldkfjalsdfjlaskjdflkas"},
             {"ano":2000, "quantidade":20, "linha_grande" : "dlsafalskjdf lflak sjdflasldfasldkfjsaldkfjalsdfjlaskjdflkas"}]
    df = pd.DataFrame(teste)

    tp = UtilPandasExcel('./teste')
    tp.write_df(df,'teste',True)

    # cor pelo ano
    str_cells = tp.range_cols(first_col=0, last_col=0,first_row=1, last_row=len(df)+1)
    tp.conditional_value_color('teste', cells= str_cells, valor='$A1=2000', cor = tp.COR_AZUL_CLARO)

    # cor pelo valor do conjunto
    str_cells = tp.range_cols(first_col=1, last_col=1,first_row=1, last_row=len(df)+1)
    tp.conditional_color('teste', cells= str_cells, min_value=0, max_value=30, mid_value=15)

    tp.header_formatting = False
    tp.write_df(df,'teste_ano',['ano'])
    tp.write_dfs({'teste2': df, 'teste3':df})


    tp.save()
    print(df)    
