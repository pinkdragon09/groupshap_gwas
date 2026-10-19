#include <R.h>
#include <Rinternals.h>
#include <stdlib.h>
#include <string.h>

/* For a fixed evaluation/background pair, a tree leaf is an indicator that
 * certain groups are present and others absent. Attribute these indicators
 * with exactly the same Shapley weights, avoiding prediction of all hybrids. */
typedef struct {
  const int *left, *right, *variable, *group, *power;
  const double *value, *background, *evaluation, *weights;
  double *phi;
  int nb, ne, evaluation_row, background_row, groups;
} Context;

static void traverse(Context *c, int node, int state) {
  if (c->left[node] == 0 && c->right[node] == 0) {
    const double *w = c->weights + state * c->groups;
    for (int g = 0; g < c->groups; ++g)
      c->phi[c->evaluation_row + c->ne * g] += c->value[node] * w[g];
    return;
  }
  int variable = c->variable[node];
  int group = c->group[variable];
  int assignment = (state / c->power[group]) % 3;
  int ev = c->evaluation[c->evaluation_row + c->ne * variable] <= c->value[node]
    ? c->left[node] : c->right[node];
  int bg = c->background[c->background_row + c->nb * variable] <= c->value[node]
    ? c->left[node] : c->right[node];
  if (assignment == 1) traverse(c, ev, state);
  else if (assignment == 2) traverse(c, bg, state);
  else if (ev == bg) traverse(c, ev, state);
  else {
    traverse(c, ev, state + c->power[group]);
    traverse(c, bg, state + 2 * c->power[group]);
  }
}

SEXP nonlinear_ld_leaf_shap(SEXP forest, SEXP background, SEXP evaluation,
                            SEXP orders, SEXP exact, SEXP mapping) {
  SEXP names = getAttrib(forest, R_NamesSymbol);
  SEXP children = R_NilValue, variables = R_NilValue, values = R_NilValue;
  for (int i = 0; i < length(forest); ++i) {
    const char *name = CHAR(STRING_ELT(names, i));
    if (!strcmp(name, "child.nodeIDs")) children = VECTOR_ELT(forest, i);
    if (!strcmp(name, "split.varIDs")) variables = VECTOR_ELT(forest, i);
    if (!strcmp(name, "split.values")) values = VECTOR_ELT(forest, i);
  }
  if (children == R_NilValue || variables == R_NilValue || values == R_NilValue)
    error("Unsupported ranger forest structure");
  int nb = INTEGER(getAttrib(background, R_DimSymbol))[0];
  int ne = INTEGER(getAttrib(evaluation, R_DimSymbol))[0];
  int groups = INTEGER(getAttrib(orders, R_DimSymbol))[1];
  int permutations = INTEGER(getAttrib(orders, R_DimSymbol))[0];
  int trees = length(children);
  if (groups < 1 || groups > 12 || nb < 1 || ne < 1 || trees < 1)
    error("Unsupported dimensions");
  int power[12], statuses[12], states = 1;
  double factorial[13]; factorial[0] = 1;
  for (int g = 0; g < groups; ++g) {
    power[g] = states; states *= 3; factorial[g + 1] = factorial[g] * (g + 1);
  }
  double *weights = (double *) R_alloc((size_t) states * groups, sizeof(double));
  memset(weights, 0, (size_t) states * groups * sizeof(double));
  for (int state = 1; state < states; ++state) {
    int present = 0, absent = 0;
    for (int g = 0; g < groups; ++g) {
      statuses[g] = (state / power[g]) % 3;
      present += statuses[g] == 1; absent += statuses[g] == 2;
    }
    if (asLogical(exact)) {
      int constrained = present + absent;
      for (int g = 0; g < groups; ++g) {
        if (statuses[g] == 1) weights[state * groups + g] =
          factorial[present - 1] * factorial[absent] / factorial[constrained];
        if (statuses[g] == 2) weights[state * groups + g] =
          -factorial[present] * factorial[absent - 1] / factorial[constrained];
      }
    } else {
      for (int p = 0; p < permutations; ++p) {
        int last = -1, last_group = -1, first = groups, first_group = -1;
        for (int position = 0; position < groups; ++position) {
          int g = INTEGER(orders)[p + permutations * position] - 1;
          if (statuses[g] == 1) { last = position; last_group = g; }
          if (statuses[g] == 2 && first == groups) { first = position; first_group = g; }
        }
        if (last < first) {
          if (last_group >= 0) weights[state * groups + last_group] += 1.0 / permutations;
          if (first_group >= 0) weights[state * groups + first_group] -= 1.0 / permutations;
        }
      }
    }
  }
  SEXP result = PROTECT(allocMatrix(REALSXP, ne, groups));
  memset(REAL(result), 0, (size_t) ne * groups * sizeof(double));
  Context c = {0};
  c.background = REAL(background); c.evaluation = REAL(evaluation);
  c.group = INTEGER(mapping); c.power = power; c.weights = weights;
  c.phi = REAL(result); c.nb = nb; c.ne = ne; c.groups = groups;
  for (int tree = 0; tree < trees; ++tree) {
    SEXP child = VECTOR_ELT(children, tree);
    SEXP left = PROTECT(coerceVector(VECTOR_ELT(child, 0), INTSXP));
    SEXP right = PROTECT(coerceVector(VECTOR_ELT(child, 1), INTSXP));
    SEXP variable = PROTECT(coerceVector(VECTOR_ELT(variables, tree), INTSXP));
    SEXP value = PROTECT(coerceVector(VECTOR_ELT(values, tree), REALSXP));
    c.left = INTEGER(left); c.right = INTEGER(right);
    c.variable = INTEGER(variable); c.value = REAL(value);
    for (int e = 0; e < ne; ++e) for (int b = 0; b < nb; ++b) {
      c.evaluation_row = e; c.background_row = b;
      traverse(&c, 0, 0);
    }
    UNPROTECT(4);
    R_CheckUserInterrupt();
  }
  for (int i = 0; i < ne * groups; ++i) REAL(result)[i] /= ((double) trees * nb);
  UNPROTECT(1);
  return result;
}
