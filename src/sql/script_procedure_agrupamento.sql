DELIMITER //  
# v_sessao: define a sessão de agrupamento - fica disponível por 12h no banco (pode ser o nick do usuário ou qualquer informação)
# v_percentual: de 70 a 100 é a similaridade mínima entre os documentos para formar grupo
# caso não seja informado v_percentual, utiliza o menos encontrado no agrupamento
CREATE OR REPLACE PROCEDURE testes.testar_grupos(v_sessao varchar(100), v_percentual int default 0) 
RETURNS query(mensagem varchar(1000) , resultado varchar(10)) AS
  DECLARE 
	arr ARRAY(RECORD(qtd INT));
	_sql text = '';
	_res text = '';
	i_percentual int = v_percentual;
  BEGIN
    /* caso o percentual não seja informado, utiliza o menor percentual do agrupamento */
	if i_percentual <=0 then
		arr = COLLECT(concat('SELECT min(sim*100) from testes.grupos where sim>0 and sessao=',quote(v_sessao)), QUERY(qtd INT));
		i_percentual = arr[0].qtd;
	end if;
	/* inicia os testes */
	_res = concat('select "Avaliando similaridade >= ', i_percentual, '" as mensagem, "ok" as resultado ');
	/* contadores */
	arr = COLLECT(concat('SELECT count(1) from testes.grupos where sessao=',quote(v_sessao)), QUERY(qtd INT));
	_res = concat(_res, 'union all select "Total de registros no grupo" as mensagem, "',arr[0].qtd,'" as resultado ');
	/* grupos */
	arr = COLLECT(concat('SELECT count(1) from testes.grupos where centroide and grupo>0 and sessao=',quote(v_sessao)), QUERY(qtd INT));
	_res = concat(_res, 'union all select "Total de grupos criados" as mensagem, "',arr[0].qtd,'" as resultado ');
    /* teste órfãos - verifica se algum órfão poderia ter sido agrupado com algum centróide */
    _sql = concat('select count(1)                                                           ',
        		  'from testes.grupos g_orfao                                                ',
        		  'inner join testes.grupos g_cent on g_cent.sessao=g_orfao.sessao           ',
        		  'and g_cent.centroide and g_cent.seq_documento <> g_orfao.seq_documento    ',
        		  'inner join testes.vetores v1 on v1.seq_documento = g_orfao.seq_documento  ', 
        		  'inner join testes.vetores v2 on v2.seq_documento = g_cent.seq_documento   ',
        		  'and dot_product(v1.vetor, v2.vetor) >= ', i_percentual/100, '             ',
        		  'where g_orfao.sessao=',quote(v_sessao),' and g_orfao.grupo=-1             ');
	arr = COLLECT(_sql, QUERY(qtd INT));
	_res = concat(_res, 'union all select "Verificação de órfãos" as mensagem, "',case when arr[0].qtd=0 then 'ok' else 'falha' end,'" as resultado ');
    /* teste de similares - verifica se algum similar poderia ter sigo agrupado com outro centróide melhor */
    _sql = concat('select count(1)                                                           ',
        		  'from testes.grupos g_sim                                                  ',
        		  'inner join testes.grupos g_cent on g_cent.sessao=g_sim.sessao             ',
        		  'and g_cent.centroide and g_cent.seq_documento <> g_sim.seq_documento      ',
        		  'and g_cent.grupo <> g_sim.grupo                                           ',
        		  'inner join testes.vetores v1 on v1.seq_documento = g_sim.seq_documento    ',
        		  'inner join testes.vetores v2 on v2.seq_documento = g_cent.seq_documento   ',
        		  'and dot_product(v1.vetor, v2.vetor) > g_sim.sim                           ',
        		  'where g_sim.sessao=', quote(v_sessao),' and g_sim.grupo>0 and not g_sim.centroide ');
	arr = COLLECT(_sql, QUERY(qtd INT));
	_res = concat(_res, 'union all select "Verificação de melhores similaridades" as mensagem, "',case when arr[0].qtd=0 then 'ok' else 'falha' end,'" as resultado ');
	/* verifica se tem mais de um centróide no mesmo grupo */
    _sql = concat('select count(1) from (                                                    ',
        		  'select grupo, count(1) from testes.grupos g                               ',
        		  'where g.grupo > 0 and centroide and sessao =', quote(v_sessao),'          ',
        		  'group by grupo                                                            ',
        		  'having count(1)>1 )                                                       ');
	arr = COLLECT(_sql, QUERY(qtd INT));
	_res = concat(_res, 'union all select "Verificação se tem só um centróide por grupo " as mensagem, "',case when arr[0].qtd=0 then 'ok' else 'falha' end,'" as resultado ');
	/* verifica se cada grupo tem um centróide */
    _sql = concat('select count(1) from testes.grupos g where grupo>0                       ',
        		  'and not exists(select 1 from testes.grupos gc where gc.grupo=g.grupo     ',
                  'and gc.centroide and sessao =', quote(v_sessao),')   ',
				  'and sessao =', quote(v_sessao));
	arr = COLLECT(_sql, QUERY(qtd INT));
	_res = concat(_res, 'union all select "Verificação se existe 1 centróide por grupo " as mensagem, "',case when arr[0].qtd=0 then 'ok' else 'falha' end,'" as resultado ');
    /* testa a similaridade dos documentos dos grupos com seus centróides */
    _sql = concat('select count(1) from testes.grupos g                                      ',
                  'inner join testes.grupos gc on gc.centroide and g.grupo =gc.grupo         ',
				  '                                            and gc.sessao =g.sessao       ',
                  'inner join testes.vetores v on v.seq_documento =g.seq_documento           ',
                  'inner join testes.vetores vc on vc.seq_documento =gc.seq_documento        ',
                  'where g.grupo >0 and g.sessao =', quote(v_sessao) ,'                      ',
				  '                 and abs(dot_product(v.vetor, vc.vetor)-g.sim)>0.01       ');
	arr = COLLECT(_sql, QUERY(qtd INT));
	_res = concat(_res, 'union all select "Verificação das similaridades calculadas" as mensagem, "',case when arr[0].qtd=0 then 'ok' else 'falha' end,'" as resultado ');
	/* retorno dos resultados dos testes */
	return TO_QUERY(_res);
  END//
DELIMITER ;


DELIMITER //  
# v_sessao: define a sessão de agrupamento - fica disponível por 12h no banco (pode ser o nick do usuário ou qualquer informação)
# v_percentual: de 70 a 100 é a similaridade mínima entre os documentos para formar grupo
CREATE OR REPLACE PROCEDURE testes.agrupar(v_sessao varchar(100), v_percentual int, qtd_batch int default 500) AS
  DECLARE 
    q int = 0;
	qtd int;
	ordemLog int = 1;
	q_max int = 50000; /* número máximo de loopings de agrupamentos */
    q_batch int = case when qtd_batch>0 then qtd_batch else 100 end; /* número de documentos comparados a cada iteração para montar a matriz */
	qry QUERY(seq_documento int) = SELECT distinct seq_documento FROM testes.grupos where sessao = v_sessao order by seq_documento;
	arrSqs ARRAY(RECORD(seq_documento int));
	arrQtd ARRAY(RECORD(qtd INT));
	tmpSq_1 int;
	tmpSq_2 int;
  BEGIN
    SET SESSION TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;
    /* prepara os logs da sessão */
	delete from testes.grupos_logs where sessao = v_sessao;
    insert into testes.grupos_logs(sessao,txt_log, ordem) values (v_sessao, concat('iniciando agrupamento com similaridade ',v_percentual), ordemLog);
	ordemLog += 1;
	/* apaga os agrupamentos antigos > 12h e atualiza o grupo atual para ficar por mais 12h */
	delete from testes.grupos g where dthr<ADDDATE(now(), INTERVAL -12 hour) and sessao <> v_sessao;
	update testes.grupos set dthr=now() where sessao=v_sessao;
	/* reinicia os números dos grupos */
	with tmp_grupo as (
	    SELECT row_number() over() AS grupo, seq_documento FROM testes.grupos where sessao=v_sessao
	)update testes.grupos g inner join tmp_grupo t on t.seq_documento=g.seq_documento
		set g.sim=null, g.grupo=t.grupo, g.centroide=false 
		where g.sessao=v_sessao;
    /* coloca como grupo -2 os registros de documento sem vetor */
	update testes.grupos g set grupo=-2 where sessao = v_sessao and 
			not exists(select 1 from testes.vetores v where v.seq_documento=g.seq_documento);
	/* atualiza o log */
	arrQtd = COLLECT(concat('select count(1) as qtd from testes.grupos where sessao=',quote(v_sessao),' and grupo=-2'), QUERY(qtd INT));
	qtd = arrQtd[0].qtd;
    insert into testes.grupos_logs(sessao,txt_log, ordem) values (v_sessao, concat('número de documentos sem vetor: ',qtd), ordemLog);
	ordemLog += 1;
	arrQtd = COLLECT(concat('select count(1) as qtd from testes.grupos where sessao=',quote(v_sessao),' and grupo<>-2'), QUERY(qtd INT));
	qtd = arrQtd[0].qtd;
    insert into testes.grupos_logs(sessao,txt_log, ordem) values (v_sessao, concat('número de documentos com vetor: ',qtd), ordemLog);
	ordemLog += 1;
	/* limpa a tabela temporária da matriz de agrupamento para sessões que não existem mais ou a sessão atual*/
	delete from testes.grupos_matriz_tmp m where m.sessao=v_sessao or not exists(select 1 from testes.grupos g where g.sessao=m.sessao);
	/* cria a matriz de similaridade para os que possuem a similaridade mínima */
	arrSqs = COLLECT(qry);
	q = length(arrSqs);
	/* faz a varredura entre os seq_documento calculando a matriz de q_batch em q_batch seq_documento ordenados
	   buscando para cada vetor todos os outros vetores similares dentro da similaridade informada	*/
	tmpSq_1 = -1;
	/* começa do segundo documento pois vai caminhando entre seqs para montar a matriz por lote */
	FOR i IN 1 .. q-1  LOOP
		tmpSq_2 = arrSqs[i].seq_documento;
		/* a cada q_max ou no último, atualiza a matriz */
		if mod(i+1,q_batch)=0 or i=q-1 then 
			insert into testes.grupos_matriz_tmp (sessao, seq_documento, seq_documento_sim, sim)
				select g1.sessao, g1.seq_documento, g2.seq_documento as seq_documento_sim, 
					case when g2.seq_documento=g1.seq_documento then 1 else DOT_PRODUCT(v1.vetor,v2.vetor) end as sim 
					from testes.grupos g1 
					inner join testes.grupos g2 on g2.sessao=v_sessao and g2.grupo<>-2 
					inner join testes.vetores v1 on v1.seq_documento = g1.seq_documento
					inner join testes.vetores v2 on v2.seq_documento = g2.seq_documento
					where g1.sessao=v_sessao and g1.grupo<>-2 
						and g1.seq_documento > tmpSq_1 and g1.seq_documento <= tmpSq_2
						and g2.seq_documento <> g1.seq_documento
						and (DOT_PRODUCT(v1.vetor,v2.vetor)>=GREATEST(70,v_percentual)/100);
			tmpSq_1 = tmpSq_2;
			/* atualiza o log */
			insert into testes.grupos_logs(sessao,txt_log, ordem) values (v_sessao, concat('preparando matriz de comparação entre documentos ',i+1,'/',q), ordemLog);
			ordemLog += 1;
		end if;
	END LOOP;
	/* atualiza o log */
	arrQtd = COLLECT(concat('select count(1) as qtd from testes.grupos_matriz_tmp where sessao=',quote(v_sessao)), QUERY(qtd INT));
	qtd = arrQtd[0].qtd;
    insert into testes.grupos_logs(sessao,txt_log, ordem) values (v_sessao, concat('matriz criada com ',qtd,' comparações entre documentos'), ordemLog);
	ordemLog += 1;
	/* finalizou a matriz de similaridade - atualiza a hora do agrupamento nos grupos
	   para postergar a remoção da sessão de agrupamento */
	update testes.grupos set dthr=now() where sessao=v_sessao;
	/* - Agrupa os documentos com os melhores pares (maior conjunto de similares) e vai removendo da matriz construída agrupada até acabarem as possibilidades 
       faz isso até não ter alteração ou no limite de q_max loopings para não ficar em loopings infinitos 
	   - no final verifica se algum documento precisa mudar para um centróide mais próximo
	   - q indica se houve alteração para que o looping continue */
	q = 1;
	/* atualiza o log */
    insert into testes.grupos_logs(sessao,txt_log, ordem) values (v_sessao, concat('iniciando criação de grupos'), ordemLog);
	ordemLog += 1;
	/* agrupa dos centróides mais densos para os menos densos */
    WHILE (q>0) and (q_max>0) LOOP
        q_max += -1;
        with centroide as (
            /* busca o centroide com mais similares próximos disponíveis */
            select seq_documento from 
            testes.grupos_matriz_tmp m where 
            sessao = v_sessao group by seq_documento
            order by count(1) desc, sum(sim) desc 
            limit 1
        ), documentos as (
        select seq_documento from testes.grupos g 
        where sessao = v_sessao
            and ( seq_documento in (select seq_documento from centroide) or 
                seq_documento in (select seq_documento_sim from testes.grupos_matriz_tmp m
                                    inner join centroide c on c.seq_documento = m.seq_documento
                                    where m.sessao = v_sessao)
                )
        ), novos as (
        select d.seq_documento, gc.grupo,
                case when m.sim is null then 1 else m.sim end as sim,
                case when m.sim is null then 1 else 0 end as centroide
                from documentos d 
        inner join centroide c on 1=1
        inner join testes.grupos gc on gc.seq_documento =c.seq_documento and gc.sessao = v_sessao
        left join testes.grupos_matriz_tmp m on m.sessao = v_sessao and 
                                m.seq_documento = c.seq_documento and 
                                m.seq_documento_sim = d.seq_documento
        ) /* atualiza os similares e o centróide do grupo atual */  
        update testes.grupos g 
        inner join novos n on n.seq_documento = g.seq_documento
        set g.grupo = n.grupo, 
            g.sim = n.sim, 
            g.centroide = n.centroide
        where g.sessao = v_sessao;

        /* verifica se foi feita alguma atualização, quando não fizer, acabou o agrupamento */
        q = row_count();

        delete from testes.grupos_matriz_tmp m where sessao = v_sessao and 
        seq_documento in (select seq_documento from testes.grupos g where g.sessao=m.sessao and g.sim is not null);

        q += row_count();
        /* se nenhuma alteração for feita, q=0 e sai */
	END LOOP;
	/* atualiza para -1 todos os documentos sem semelhantes*/
	update testes.grupos g set grupo=-1, sim=0, centroide=false where sessao=v_sessao and sim is null;
	/* atualiza logs */
	arrQtd = COLLECT(concat('select count(1) as qtd from testes.grupos where sessao=',quote(v_sessao),' and grupo>0 and centroide'), QUERY(qtd INT));
	qtd = arrQtd[0].qtd;
    insert into testes.grupos_logs(sessao,txt_log, ordem) values (v_sessao, concat('reorganizando documentos pelo centróide mais próximo para ',qtd,' centróides(s)'), ordemLog);
	ordemLog += 1;
	/* verifica se algum documento poderia ter sido agrupado por um centróide mais próximo */
	with ajustes as (
		select g_sim.seq_documento, g_sim.grupo, g_sim.sim, g_cent.seq_documento as seq_documento_cent, 
			g_cent.grupo as grupo_cent, dot_product(v1.vetor, v2.vetor) as sim_cent
		from testes.grupos g_sim
		inner join testes.grupos g_cent on g_cent.sessao=g_sim.sessao 
				and g_cent.centroide and g_cent.seq_documento <> g_sim.seq_documento
				and g_cent.grupo <> g_sim.grupo
		inner join testes.vetores v1 on v1.seq_documento = g_sim.seq_documento 
		inner join testes.vetores v2 on v2.seq_documento = g_cent.seq_documento 
				and dot_product(v1.vetor, v2.vetor) > g_sim.sim 
		where g_sim.sessao=v_sessao and g_sim.grupo>0 and not g_sim.centroide 
	), max_ajustes as (
		select * from ajustes a 
		where not exists(select 1 from ajustes a2 where a2.seq_documento = a.seq_documento and 
		  				 a2.sim_cent>a.sim_cent)
	) /* atualiza os similares para o centróide mais próximo */
		update testes.grupos g 
			inner join max_ajustes aj on aj.seq_documento = g.seq_documento 
			set g.sim = aj.sim_cent, g.grupo = aj.grupo_cent
		where not (g.centroide) and g.grupo >0 and g.grupo <> aj.grupo_cent
			and g.sessao=v_sessao;
	/* atualiza para -1 todos os documentos sem semelhantes - algum centróide pode ter ficado órfão */
	update testes.grupos g set grupo=-1, sim=0, centroide=false where sessao=v_sessao and
	       not exists(select 1 from testes.grupos g2 where g2.seq_documento<>g.seq_documento and g2.grupo=g.grupo and sessao=v_sessao);
	/* renumera os grupos iniciando de 1 */
	update testes.grupos g set g.grupo=g.grupo+(select max(t.grupo) from testes.grupos t 
			where t.sessao=g.sessao)+1000000 where g.sessao=v_sessao and g.grupo>=0;
	with tmpqtd as (
		select count(1) as qtd, grupo from testes.grupos where sessao=v_sessao group by grupo
	),	tmpnum as (
		select row_number() over(order by qt.qtd desc, g.grupo) as novo_grupo, g.grupo, g.sessao, qt.qtd from testes.grupos g
		inner join tmpqtd qt on qt.grupo=g.grupo
		where g.sessao=v_sessao and g.grupo>=0 
		group by g.grupo 
	)update testes.grupos g inner join tmpnum t 
			set g.grupo=t.novo_grupo where g.sessao=t.sessao and g.grupo=t.grupo;
	/* limpa as tabelas de agrupamento */
	delete from testes.grupos_matriz_tmp where sessao=v_sessao;
	/* finaliza o log removendo os com mais de 30 dias */
	delete from testes.grupos_logs where dthr<ADDDATE(now(), INTERVAL -30 day) and sessao <> v_sessao;
	arrQtd = COLLECT(concat('select count(1) as qtd from testes.grupos where sessao=',quote(v_sessao),' and grupo>0 and centroide'), QUERY(qtd INT));
	qtd = arrQtd[0].qtd;
    insert into testes.grupos_logs(sessao,txt_log, ordem) values (v_sessao, concat('finalizado agrupamento com ',qtd,' grupo(s) criado(s)'), ordemLog);
  END//
DELIMITER ;
