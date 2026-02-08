"""Sample products and initial prompts for the evaluation PoC."""

SAMPLE_PRODUCTS = [
    {
        "name": "Smartphone Samsung Galaxy S24",
        "category": "Eletrônicos > Celulares e Smartphones",
        "description": (
            "Smartphone Samsung Galaxy S24 256GB Preto 5G "
            "- Octa-Core 8GB RAM 6,2\" Câm. Tripla + Selfie 12MP"
        ),
        "brand": "Samsung",
        "attributes": {
            "cor": "Preto",
            "armazenamento_interno": "256GB",
        },
        "expected_attributes": {
            "cor": "Preto",
            "armazenamento_interno": "256GB",
            "memoria_ram": "8GB",
            "tamanho_tela": "6.2 polegadas",
            "tipo_tela": "Dynamic AMOLED 2X",
            "resolucao_tela": "2340 x 1080 (FHD+)",
            "processador": "Snapdragon 8 Gen 3",
            "camera_traseira": "50MP + 12MP + 10MP",
            "camera_frontal": "12MP",
            "bateria": "4000mAh",
            "sistema_operacional": "Android 14",
            "conectividade": "5G",
            "peso": "167g",
            "resistencia_agua": "IP68",
            "dual_sim": "Sim (Nano SIM + eSIM)",
        },
    },
    {
        "name": "Cafeteira Nespresso Vertuo Next",
        "category": "Eletrodomésticos > Cafeteiras",
        "description": (
            "Cafeteira Nespresso Vertuo Next Preta com Cápsulas - 1260W"
        ),
        "brand": "Nespresso",
        "attributes": {
            "cor": "Preta",
            "tipo": "Cápsulas",
        },
        "expected_attributes": {
            "cor": "Preta",
            "tipo": "Cápsulas",
            "potencia": "1260W",
            "voltagem": "110V/220V",
            "pressao": "19 bar",
            "capacidade_reservatorio": "1.1L",
            "tipo_capsula": "Vertuo",
            "material": "Plástico reciclado",
            "dimensoes": "14.2 x 42.9 x 31.4 cm",
            "peso": "4kg",
            "tempo_aquecimento": "30 segundos",
            "desligamento_automatico": "Sim, após 9 minutos",
            "tamanhos_bebida": (
                "Espresso (40ml), Double Espresso (80ml), "
                "Gran Lungo (150ml), Mug (230ml), Alto (414ml)"
            ),
        },
    },
    {
        "name": "Tênis Nike Air Max 90",
        "category": "Moda > Calçados > Tênis",
        "description": "Tênis Nike Air Max 90 Masculino - Branco e Preto",
        "brand": "Nike",
        "attributes": {
            "cor": "Branco e Preto",
            "genero": "Masculino",
        },
        "expected_attributes": {
            "cor": "Branco e Preto",
            "genero": "Masculino",
            "material_cabedal": "Couro e mesh",
            "material_solado": "Borracha",
            "tecnologia_amortecimento": "Air Max (câmara de ar visível)",
            "tipo_pisada": "Neutra",
            "fechamento": "Cadarço",
            "altura_cano": "Baixo",
            "indicado_para": "Casual / Lifestyle",
            "peso_aproximado": "340g (tamanho 42)",
            "origem": "Importado",
        },
    },
]

INITIAL_PROMPTS = [
    {
        "id": "prompt_v1",
        "name": "Prompt Simples (v1)",
        "template": (
            "Você é um especialista em catálogo de produtos. "
            "Dado o produto abaixo, enriqueça a ficha técnica com todos "
            "os atributos relevantes.\n\n"
            "Produto: {product_name}\n"
            "Categoria: {category}\n"
            "Descrição: {description}\n"
            "Marca: {brand}\n"
            "Atributos existentes: {attributes}\n\n"
            "Retorne APENAS um JSON com todos os atributos enriquecidos "
            "(incluindo os existentes)."
        ),
        "rationale": "Prompt direto e simples, sem exemplos ou formato específico.",
    },
    {
        "id": "prompt_v2",
        "name": "Prompt Estruturado (v2)",
        "template": (
            "Você é um especialista em enriquecimento de fichas técnicas "
            "de produtos para e-commerce.\n\n"
            "## Tarefa\n"
            "Analise o produto abaixo e enriqueça sua ficha técnica com o "
            "máximo de atributos relevantes para a categoria.\n\n"
            "## Produto\n"
            "- Nome: {product_name}\n"
            "- Categoria: {category}\n"
            "- Descrição: {description}\n"
            "- Marca: {brand}\n"
            "- Atributos atuais: {attributes}\n\n"
            "## Instruções\n"
            "1. Mantenha todos os atributos existentes\n"
            "2. Adicione atributos técnicos relevantes para a categoria\n"
            "3. Use valores específicos e precisos "
            "(evite \"N/A\" ou valores genéricos)\n"
            "4. Inclua unidades de medida quando aplicável\n"
            "5. Considere atributos que um consumidor buscaria ao "
            "comparar produtos\n\n"
            "## Formato de saída\n"
            "Retorne APENAS um objeto JSON válido com os atributos como "
            "chave-valor. Use snake_case para as chaves em português."
        ),
        "rationale": (
            "Prompt com instruções detalhadas, regras claras e "
            "formato de saída bem definido."
        ),
    },
]
