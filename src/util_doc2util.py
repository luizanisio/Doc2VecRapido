# -*- coding: utf-8 -*-

import os
import hashlib
#######################################################################
# Arquivo de apoio para o Doc2VecRapido e o BertRapido 
# Esse código, dicas de uso e outras informações: 
#   -> https://github.com/luizanisio/Doc2VecRapido
# Luiz Anísio 
# 10/03/2023 - disponibilizado no GitHub  
#######################################################################

#####################################
from multiprocessing.dummy import Pool as ThreadPool
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from multiprocessing import cpu_count
from tqdm import tqdm
import regex as re

class UtilDocs():
    N_THREADS_PADRAO = cpu_count() * 3
    
    @classmethod
    def hash(cls, texto):
        hash_object = hashlib.sha1(str(texto).encode())
        return f'{hash_object.hexdigest()}'

    @classmethod
    def progress_bar(cls, current_value, total, msg=''):
        increments = 25
        percentual = int((current_value / total) * 100)
        i = int(percentual // (100 / increments))
        text = "\r[{0: <{1}}] {2:.2f}%".format('=' * i, increments, percentual)
        print('{} {}           '.format(text, msg), end="\n" if percentual == 100 else "")

    @classmethod
    def map_thread(cls, func, lista, n_threads=N_THREADS_PADRAO, tdqm_progress = True):
        # print('Iniciando {} threads'.format(n_threads))
        _qtd = [len(lista)]
        progresso = Progresso(_qtd[0]) if tdqm_progress else None
        def __func_progress__(v):
            _qtd[0] -= 1
            progresso.restantes(_qtd[0])
            func(v)
        pool = ThreadPool(n_threads)
        if tdqm_progress:
           pool.map(__func_progress__, lista)
           progresso.close()
        else:
           pool.map(func, lista)
        pool.close()
        pool.join()  

    @classmethod
    def any_with_threads(cls, func, lista, n_threads=N_THREADS_PADRAO):
        ''' func vai retornar o valor procurado, interrompendo assim que achar ele
            caso func retorne None, vai continuar procurando
            ao final retorna None caso não encontre o valor  '''
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = {executor.submit(func, item) for item in lista}
            for future in concurrent.futures.as_completed(futures):
                if future.result():
                   return future.result()
        return False

    @classmethod
    def apply_threads(cls, func, lista, n_threads=N_THREADS_PADRAO):
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            resultados = executor.map(func, lista)
        return resultados

    @classmethod
    def printlog(cls, inicio,msg, retornar_msg_erro=False, destaque = False):
        msg = f'> {inicio}: {msg}'
        if retornar_msg_erro:
            return msg
        if destaque: 
           cls.print_linha()
        print(msg)
        if destaque: 
           cls.print_linha()

    @classmethod
    def print_linha(cls, caractere='='):
        print(str(caractere * 70)[:70])

    # carrega os documentos txt de uma pasta e usa como tag o que estiver depois da palavra tags separadas por espaço
    # exemplo: 'arquivo texto 1 tags a b c.txt' tags será igual a ['a', 'b', 'c']
    @classmethod
    def carregar_documentos(cls, pasta, msg_log = 'UtilDocs'):
        cls.printlog(msg_log, f'Carregando documentos da pasta "{pasta}"')
        arquivos = cls.listar_arquivos(pasta)
        cls.printlog(msg_log, f'Documentos encontrados: {len(arquivos)}')
        documentos = []
        def _func(i):
            arquivo = arquivos[i]
            documento = cls.carregar_arquivo(arquivo)
            rotulo = os.path.split(arquivo)[1]            
            rotulo = os.path.splitext(rotulo)[0].lower()
            # se encontrar tag ou tags no nome do arquivo, busca as tags
            if rotulo.find('tags ') >=0:
               rotulos = rotulo.split('tags ')[1].strip().split(' ')
               print(f'Rótulo encontrado: {arquivo} >> tags: {rotulos}')
            elif rotulo.find('tag ') >=0:
               rotulos = rotulo.split('tag ')[1].strip().split(' ')
               print(f'Rótulo encontrado: {arquivo} >> tags: {rotulos}')
            else:
               rotulos = [rotulo.strip()]
            documentos.append( (documento, rotulos) )
        cls.map_thread(_func,list(range(len(arquivos))) )
        cls.printlog(msg_log, f'Documentos carregados: {len(documentos)}')
        #documentos, rotulos = list(zip(*documentos))
        #cls.printlog(f'Exemplo de documento: {documentos[0]} e rótulos: {rotulos[0]}')
        if len(documentos) == 0:
            return [],[]
        return list(zip(*documentos))

    # função simples de carga de arquivos que tenta descobrir o tipo de arquivo (utf8, ascii, latin1)
    @classmethod
    def carregar_arquivo(cls, arquivo):
        tipos = ['utf8', 'ascii', 'latin1']
        texto = None
        for tp in tipos:
            try:
                with open(arquivo, encoding=tp) as f:
                    texto = f.read()
                    break
            except UnicodeError:
                continue
        if not texto:
            with open(arquivo, encoding='latin1', errors='ignore') as f:
                texto = f.read()
        return texto.replace('<br>','\n') 

    @classmethod
    def listar_arquivos(cls, pasta, extensao='txt'):
        if not os.path.isdir(pasta):
            msg = f'Não foi encontrada a pasta "{pasta}" para listar os arquivos "{extensao}"'
            raise Exception(msg)
        res = []
        _extensao = f".{extensao}".lower() if extensao else ''
        for path, _, file_list in os.walk(pasta):
            for file_name in file_list:
                if file_name.lower().endswith(f"{_extensao}"):
                   res.append(os.path.join(path,file_name))
        return res

    @classmethod    
    def is_iterable(self, obj):
        try:
            iter(obj)
            return True
        except TypeError:
            return False 

    @classmethod
    def progresso_continuado(self, mensagem_inicial = None, i = None, total = None):
        '''Usar:
           progresso_continuado('Iniciando looping')
           progresso_continuado(i=10, total=100)
        '''
        if mensagem_inicial is None and i is None and total is None:
            return    
        if mensagem_inicial != None:
            print(f'{mensagem_inicial}', flush=True)
        if i is None:
            return
        if   total < 500:   salto = 10
        elif total < 1000:  salto = 100
        elif total < 10000: salto = 250
        else: salto = 500
        if i % salto == 0:
           print(f'{i}|', end = '', flush=True)   

    SEPARADORES = ['\n','\r', '.', '?', '!', ';', ',', ':', ' '] 
    RE_QUEBRA_LINHAS = re.compile(r'[^\n\r]+')

    
    @classmethod
    def quebrar_linhas(cls, texto, id_doc = ''):
        chunks = []

        for chunk in cls.RE_QUEBRA_LINHAS.finditer(texto):
            start = chunk.start()
            end = chunk.end()
            match_text = chunk.group()
            chunks.append({
                'inicio': start,
                'fim': end,
                'tamanho': end - start,
                'trecho': match_text,
                'id_doc': id_doc
            })

        return chunks        
    
    @classmethod
    def quebrar_pedacos(cls, texto, tamanho=500, sobreposicao=100, tolerancia=50, id_doc = '', incluir_trecho = False):
        # prepara as quebras
        chunks = []
        start = 0
        end = tamanho
        sobreposicao = sobreposicao if sobreposicao < tamanho else sobreposicao-1
        if len(texto) <= tamanho + tolerancia:
            return []

        while start < len(texto):
            # não tem o que fazer, o restante do texto é menor que o chunck + tolerancia
            if end + tolerancia >= len(texto):
                #print(f'--- Chegou ao final pela a tolerancia: start:end={start}:{end} tolerancia={tolerancia} len = {len(texto)} "{texto[start:end]}"',)
                end = len(texto)
                dict_chunk = {'inicio': start, 'fim': end, 'tamanho': end-start, 'id_doc': id_doc}
                dict_chunk['trecho'] = texto[start:end] if incluir_trecho else None
                chunks.append(dict_chunk)
                break

            if tolerancia > 0:
               for separador in cls.SEPARADORES:
                   #print(f'--- Buscando separador no final "{separador}" => "{texto[start:end]}"')
                   _ok = False
                   for i in range(tolerancia):
                       if texto[end+i-1] == separador:
                           end = end+i-1
                           _ok = True 
                           break
                   if _ok:
                       #print(f'--- Separador final encontrado "{separador}" => "{texto[start:end]}"')
                       break  

            dict_chunk = {'inicio': start, 'fim': end, 'tamanho': end-start, 'id_doc': id_doc}
            dict_chunk['trecho'] = texto[start:end] if incluir_trecho else None
            chunks.append(dict_chunk)

            # próximo chunck
            if end >= len(texto):
               # não precisa de overlap, pois chegou no final
               #print('--- Chegou no final - ignorando overlap') 
               break
            start = end - sobreposicao
            end = start + tamanho
            # verifica se pode ajustar o início para os separadores conhecidos dentro do overlap de caracteres
            if sobreposicao > 0:
               for separador in cls.SEPARADORES:
                   # print(f'--- Buscando separador no início "{separador}" => "{texto[start:end]}"')
                   _ok = False
                   for i in range(sobreposicao):
                       if texto[start + i -1] == separador:
                          start += i
                          end = start + tamanho
                          _ok = True 
                          break
                   if _ok:
                      #print(f'--- Separador inicial encontrado "{separador}" => "{texto[start:end]}"')
                      break  
        return chunks


    @classmethod
    def unir_paragrafos_ocr(self, texto):
        lista = texto if type(texto) is list else texto.split('\n')
        res = []
        def _final_pontuacao(_t):
            if len(_t.strip()) == 0:
                return False
            return _t.strip()[-1] in PONTUACAO_FINAL_LISTA
        for i, linha in enumerate(lista):
            # print('linha {}: |{}| '.format(i,linha.strip()), _final_pontuacao(linha), )
            if (i>0) and (not _final_pontuacao(lista[i-1])) or \
                (_final_pontuacao(lista[i-1]) and (ABREVIACOES_RGX.search(lista[i-1]))):
                # print('juntar: ', lista[i-1].strip(), linha.strip())
                if len(res) ==0: res =['']
                res[len(res)-1] = res[-1].strip() + ' '+ linha
            else:
                res.append(linha)
        return '\n'.join(res)

    @classmethod
    def testar(cls):
        texto = 'linha 1\nlinha2\n\noutra linha\nlinha final'
        print(cls.quebrar_linhas(texto))

#############################################################
# para a correção de quebras de texto, principalmente de extração de PDFs
ABREVIACOES = ['sra?s?', 'exm[ao]s?', 'ns?', 'nos?', 'doc', 'ac', 'publ', 'ex', 'lv', 'vlr?', 'vls?',
               'exmo\(a\)', 'ilmo\(a\)', 'av', 'of', 'min', 'livr?', 'co?ls?', 'univ', 'resp', 'cli', 'lb',
               'dra?s?', '[a-z]+r\(as?\)', 'ed', 'pa?g', 'cod', 'prof', 'op', 'plan', 'edf?', 'func', 'ch',
               'arts?', 'artigs?', 'artg', 'pars?', 'rel', 'tel', 'res', '[a-z]', 'vls?', 'gab', 'bel',
               'ilm[oa]', 'parc', 'proc', 'adv', 'vols?', 'cels?', 'pp', 'ex[ao]', 'eg', 'pl', 'ref',
               '[0-9]+', 'reg', 'f[ilí]s?', 'inc', 'par', 'alin', 'fts', 'publ?', 'ex', 'v. em', 'v.rev',
               'des', 'des\(a\)', 'desemb']
ABREVIACOES_RGX = re.compile(r'(?:{})\.\s*$'.format('|\s'.join(ABREVIACOES)), re.IGNORECASE)
PONTUACAO_FINAL_LISTA = {'.','?','!'}
#############################################################


class Progresso():
    '''Progresso com tqdm para usar em threads quando 
       se sabe apenas quantos faltam para acabar'''
    def __init__(self, total):
        self.num_itens_total = total
        self.pbar = tqdm(total=self.num_itens_total)
    
    def update(self, n=1):
        self.pbar.update(n)
        
    def restantes(self, n):
        num_itens_processados = self.num_itens_total - n
        self.pbar.update(num_itens_processados - self.pbar.n)            
        
    def close(self):
        self.pbar.close()
        