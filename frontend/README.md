# Insurance Fraud Detector — Vue Frontend

SPA em **Vue 3 + Vite + TypeScript + Tailwind CSS** que consome a API FastAPI de análise de sinistros.

## Stack

| Camada | Tecnologia |
|---|---|
| Framework | Vue 3 (Composition API + `<script setup>`) |
| Build | Vite 6 |
| Tipagem | TypeScript strict |
| Estilos | Tailwind CSS 3 |
| Routing | Vue Router 4 |
| Estado | Composable com estado partilhado no módulo (`useCases`) |
| HTTP | `fetch` nativo — sem Axios |

## Estrutura

```
src/
├── api/cases.ts          ← Cliente HTTP tipado (todos os endpoints)
├── composables/
│   └── useCases.ts       ← Estado partilhado da lista de casos
├── types/index.ts         ← Interfaces TypeScript do contrato OpenAPI
├── router/index.ts
├── views/
│   ├── EmptyView.vue      ← Estado vazio (rota /)
│   ├── CaseDetailView.vue ← Detalhe, upload, análise
│   └── NewCaseView.vue    ← Formulário de criação
├── components/
│   ├── AnalysisResult.vue ← Veredicto, probabilidade, reasoning, incongruências
│   ├── ImageGallery.vue   ← Grelha de fotos com tags YOLO
│   ├── StatementCard.vue  ← Depoimento individual
│   ├── AppSpinner.vue
│   └── AppAlert.vue
└── App.vue               ← Layout split: sidebar (lista) + RouterView
```

## Configuração

```bash
cp .env.example .env
```

Edita `VITE_API_BASE_URL` com o URL da API (ex: `http://localhost:8000`).

Em **desenvolvimento**, o proxy do Vite encaminha `/cases` e `/static` automaticamente, por isso podes deixar vazio ou apontar para o porto local.

## Correr localmente

```bash
npm install
npm run dev
# → http://localhost:5173
```

## Build para produção

```bash
npm run build
# output em dist/
```

Serve o conteúdo de `dist/` com qualquer servidor estático (nginx, Caddy, etc.) e certifica-te que a API está acessível em `VITE_API_BASE_URL`.

## Fluxo de uso

1. **Lista de casos** — sidebar carrega `GET /cases` ao abrir a app
2. **Ver detalhe** — clica num caso → `GET /cases/{id}` → depoimentos, fotos, resultado da análise
3. **Upload de fotos** — arrasta/seleciona imagens → `POST /cases/{id}/images`
4. **Analisar** — botão "Run Analysis" → `POST /cases/{id}/analyze` → resultado aparece inline
5. **Novo caso** — botão "New Case" → formulário → `POST /cases` → redireciona para detalhe

## Endpoints consumidos

| Método | Path | Uso |
|---|---|---|
| `GET` | `/cases` | Listar casos na sidebar |
| `POST` | `/cases` | Criar novo caso |
| `GET` | `/cases/{id}` | Carregar detalhe |
| `POST` | `/cases/{id}/images` | Upload de fotos |
| `POST` | `/cases/{id}/analyze` | Correr modelo Qwen3 |
