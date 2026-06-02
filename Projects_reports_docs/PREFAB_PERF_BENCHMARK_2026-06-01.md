# Prefab UI — Performance Benchmark

**Date** : 2026-06-01
**Serveur** : Pulse HTTP (`:8888`), cible locale
**Mesures** : 3 runs par dashboard (moyenne)

## Dashboard Tree Complexity

Classement par nombre de nœuds Prefab dans l'arbre `structuredContent.view` :

| Dashboard | Nœuds | Profondeur | Types dominants | Payload |
|-----------|-------|------------|-----------------|---------|
| `pulse_dashboard` | **273** | 18 | Muted(47), Row(30), Column(28), DataTable(18), Badge(8) | 38.5 KB |
| `ctf_dashboard` | **65** | 14 | Text(14), Row(9), Badge(8), Card(8) | 4.1 KB |
| `pentest_report` (empty) | **21** | 6 | Metric(4), Row(2), Alert(2) | 1.1 KB |
| `recon_summary` (empty) | **18** | 6 | Metric(4), Row(2) | 0.9 KB |

## Response Time

Le temps côté serveur (génération du `structuredContent` + sérialisation) :

| Dashboard | Temps moyen | Min / Max |
|-----------|-------------|-----------|
| `pulse_dashboard` | **604 ms** | 580 / 625 |
| `pentest_report` (empty) | **141 ms** | 132 / 148 |
| `ctf_dashboard` | **136 ms** | 129 / 142 |
| `recon_summary` (empty) | **77 ms** | 72 / 81 |

> **Note** : `pentest_report` et `recon_summary` sont mesurés "à vide" (pas de cache). Avec données réelles (scan préalable), le temps augmente de ~50-100ms (plus de nœuds). `pulse_dashboard` agrège 18 data sources en parallèle — c'est le cas nominal.

## Payload Size (SSE)

La taille du message SSE complet (incluant l'enveloppe JSON-RPC) :

| Dashboard | Bytes | Temps transfert (est.) |
|-----------|-------|----------------------|
| `pulse_dashboard` | 38 502 | ~1.5 ms @ 1 Gbps |
| `ctf_dashboard` | 4 121 | ~0.2 ms |
| `pentest_report` (empty) | 1 064 | ~0.05 ms |
| `recon_summary` (empty) | 894 | ~0.05 ms |

Temps réseau négligeable (<2ms) — le goulot est la génération serveur, pas la bande passante.

## Détail par Dashboard

### 1. `ctf_dashboard` — 136ms / 4.1KB / 65 nœuds

```
Icon(swords) → Heading → Badge(7 categories)
Separator
Row
├── Column (flex-1)
│   ├── Muted("Categories & Tools")
│   └── Card × 7 (crypto, web, forensics, osint, ...)
│       └── CardContent
│           ├── Badge(cat)
│           └── Text(tools)
└── Column (w-64)
    ├── Muted("Tool Coverage")
    └── Card
        └── CardContent
            └── BarChart(7 bars)
```

8 Badge, 8 Card, 1 BarChart. Structure plate — pas de récursion profonde.

### 2. `pentest_report` (vide) — 141ms / 1.1KB / 21 nœuds

Quand pas de findings : fallback `Alert` + `Metric` × 4.

Avec findings (post-scan) : structure attendue ~80 nœuds avec `Accordion` par sévérité, `Code` blocks pour exploits, `Table` pour ports.

### 3. `recon_summary` (vide) — 77ms / 0.9KB / 18 nœuds

Plus rapide — 4 `Metric` en `Row`. Avec cache : badges technos + DataTable historique.

### 4. `pulse_dashboard` — 604ms / 38.5KB / 273 nœuds

Le dashboard principal agrège 18 data sources en parallèle (`asyncio.gather`) :

```
Header (Tooltip × 3, Sparkline × 2, Progress × 2)
Scope Bar
Tabs
├── Overview → 7 panels (Status, Resource Gauges, Surface, Findings, 
│               System Trends, Cache, Intelligence)
├── Workflow → 5 panels (Plan IDE, Active Tools, Async Scans, 
│               Missing Tools, Rate Limit)
└── History → 6 panels (History, Sessions, Errors, Performance, 
               Confirmations, Network I/O)
Footer
```

18 `DataTable`, 47 `Muted` labels, 30 `Row`, 28 `Column`. Arbre 18 niveaux de profondeur.

## Analyse

### Temps serveur

| Seuil | Dashboard | Verdict |
|-------|-----------|---------|
| <100ms | recon_summary, pentest_report (empty) | ✅ Excellent |
| 100-200ms | ctf_dashboard, pentest_report (with data) | ✅ Bon |
| 200-500ms | — | — |
| 500ms-1s | pulse_dashboard | ⚠️ Correct (18 sources parallèles) |
| >1s | — | 🚫 Aucun |

### Complexité Prefab

Tous les dashboards tiennent dans la limite Prefab. Le plus complexe (`pulse_dashboard` avec 273 nœuds) reste sous le seuil où Claude Desktop commence à ralentir (~500+ nœuds selon les tests communauté).

### Suggestion d'optimisation

- `pulse_dashboard` → 604ms : le goulot est la collecte 18 data sources, pas le rendu Prefab. Déjà parallélisé (`asyncio.gather`). Si nécessaire : lazy loading des tabs (Overview d'abord, Workflow/History après). Non prioritaire.

## Résumé

```
              Payload     Temps    Nœuds
ctf_dashboard    4 KB    136 ms      65  ✅
pentest_report   1 KB    141 ms      21  ✅  (~80 avec données)
recon_summary    1 KB     77 ms      18  ✅  (~50 avec données)
pulse_dashboard  39 KB   604 ms     273  ⚠️
```

**Aucun dashboard ne pose problème de performance.** Le plus lent est `pulse_dashboard` à 604ms, acceptable pour un tableau de bord qui agrège 18 sources en temps réel.
