"""
Testes unitários de learn.py (fora dos arquivos de entrega).

Cobertura atual:
    Etapa 1 - initialize_parameters, relu, softmax (+ correção do OneHotEncoder)
    Etapa 2 - forward
    Etapa 3 - cross_entropy_loss, accuracy

Como rodar (a partir de codigo-base/):
    .venv/bin/python tests/test_learn.py
Também é compatível com pytest, caso instalado:
    pytest tests/test_learn.py -v
"""
import math
import os
import sys

import numpy as np

# Permite importar learn.py, que está em codigo-base/src/
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC_DIR)
import learn  # noqa: E402

ATOL = 1e-8  # tolerância para comparar com valores impressos no PDF (8 casas)


# ---------------------------------------------------------------------------
# Valores esperados copiados do enunciado (tp2-rna.pdf)
# ---------------------------------------------------------------------------

# Seção 3.1 - initialize_parameters(784, 4, 10) com semente 42
PDF_W1_FIRST3 = np.array([
    [ 0.00496714, -0.00138264,  0.00647689,  0.0152303 ],
    [-0.00234153, -0.00234137,  0.01579213,  0.00767435],
    [-0.00469474,  0.0054256 , -0.00463418, -0.0046573 ],
])
PDF_W1_LAST3 = np.array([
    [-0.00139397,  0.00739653, -0.01909356,  0.01318302],
    [ 0.00072681, -0.00411493, -0.00089234, -0.00037571],
    [-0.01731201,  0.01494938,  0.00041326,  0.00443002],
])
PDF_W2 = np.array([
    [ 0.0095155 , -0.01021162,  0.00473472, -0.00267641,  0.00846771, -0.02127227,
     -0.00099094, -0.00602821,  0.00432263,  0.00470044],
    [-0.00707626, -0.00712174, -0.00110665, -0.00896642,  0.00841984, -0.00369207,
     -0.02906988, -0.00374822, -0.01038544, -0.01631276],
    [-0.01237427,  0.00109408,  0.01328641,  0.00313184, -0.00606503,  0.00455904,
     -0.0045909 , -0.006946  , -0.01154363, -0.01751829],
    [-0.00389924,  0.00158053, -0.00096624, -0.00415967, -0.00945746,  0.00608246,
     -0.01317132,  0.00776028, -0.01002217, -0.00752435],
])

# Seção 3.2 - funções de ativação
PDF_SOFTMAX_ROW = np.array([0.09003057, 0.24472847, 0.66524096])

# Seção 3.3 - forward com X = np.random.randn(2, 784) logo após a inicialização
PDF_Z1 = np.array([
    [ 0.24242511, -0.1806328 , -0.32022768,  0.05051037],
    [-0.3853982 , -0.13327416,  0.02946697,  0.13882065],
])
PDF_A1 = np.array([
    [0.24242511, 0.        , 0.        , 0.05051037],
    [0.        , 0.        , 0.02946697, 0.13882065],
])
PDF_Z2 = np.array([
    [ 0.00210984, -0.00239572,  0.00109901, -0.00085894,  0.00157509, -0.00484971,
     -0.00090552, -0.00106941,  0.00054169,  0.00075945],
    [-0.00090593,  0.00025165,  0.00025738, -0.00048516, -0.00149161,  0.00097871,
     -0.00196373,  0.00087261, -0.00173144, -0.00156075],
])
PDF_A2 = np.array([
    [0.10025104, 0.09980037, 0.10014976, 0.09995386, 0.10019745, 0.09955576,
     0.09994921, 0.09993283, 0.10009396, 0.10011576],
    [0.09996714, 0.10008293, 0.1000835 , 0.10000921, 0.09990861, 0.10015572,
     0.09986145, 0.10014509, 0.09988465, 0.0999017 ],
])


# ---------------------------------------------------------------------------
# Auxiliares
# ---------------------------------------------------------------------------

def pdf_setup():
    """Reproduz a sequência do PDF: semente 42 -> inicialização -> X aleatório.

    A semente é refixada aqui porque learn.py só a fixa uma vez, na importação;
    sem isso o resultado dependeria da ordem em que os testes rodam.
    """
    np.random.seed(42)
    parameters = learn.initialize_parameters(784, 4, 10)
    X = np.random.randn(2, 784)
    return parameters, X


def random_parameters(rng, n, H, C):
    """Parâmetros aleatórios de escala ~1 (maiores que os da inicialização),
    para que as diferenças entre implementações apareçam nos testes."""
    return (rng.normal(size=(n, H)), rng.normal(size=(1, H)),
            rng.normal(size=(H, C)), rng.normal(size=(1, C)))


def naive_forward(X, parameters):
    """Forward escrito elemento a elemento, só com laços e math.exp,
    seguindo literalmente as fórmulas da Seção 3.3. Serve de referência."""
    W1, b1, W2, b2 = parameters
    m, n = X.shape
    H, C = W1.shape[1], W2.shape[1]
    Z1, A1, Z2, A2 = (np.zeros((m, H)), np.zeros((m, H)),
                      np.zeros((m, C)), np.zeros((m, C)))
    for i in range(m):
        for j in range(H):
            Z1[i, j] = sum(X[i, k] * W1[k, j] for k in range(n)) + b1[0, j]
            A1[i, j] = max(0.0, Z1[i, j])
        for j in range(C):
            Z2[i, j] = sum(A1[i, k] * W2[k, j] for k in range(H)) + b2[0, j]
        exps = [math.exp(Z2[i, j]) for j in range(C)]
        total = sum(exps)
        for j in range(C):
            A2[i, j] = exps[j] / total
    return Z1, A1, Z2, A2


# ===========================================================================
# ETAPA 1 - initialize_parameters, relu, softmax
# ===========================================================================

def test_etapa1_init_retorna_tupla_com_shapes_corretos():
    params = learn.initialize_parameters(784, 4, 10)
    assert isinstance(params, tuple) and len(params) == 4
    W1, b1, W2, b2 = params
    assert W1.shape == (784, 4)
    assert b1.shape == (1, 4)    # viés 2D, não (4,)
    assert W2.shape == (4, 10)
    assert b2.shape == (1, 10)   # viés 2D, não (10,)


def test_etapa1_init_valores_do_pdf():
    np.random.seed(42)
    W1, b1, W2, b2 = learn.initialize_parameters(784, 4, 10)
    assert np.allclose(W1[:3], PDF_W1_FIRST3, atol=ATOL)
    assert np.allclose(W1[-3:], PDF_W1_LAST3, atol=ATOL)
    assert np.allclose(W2, PDF_W2, atol=ATOL)


def test_etapa1_init_vieses_zerados():
    _, b1, _, b2 = learn.initialize_parameters(784, 4, 10)
    assert np.all(b1 == 0) and np.all(b2 == 0)


def test_etapa1_init_pesos_quebram_simetria():
    # Nenhuma coluna (neurônio) de W pode ser igual a outra
    W1, _, W2, _ = learn.initialize_parameters(784, 16, 10)
    assert len(np.unique(W1, axis=1).T) == W1.shape[1]
    assert len(np.unique(W2, axis=1).T) == W2.shape[1]


def test_etapa1_init_escala_aproximada_0_01():
    # randn * 0.01 -> média ~0 e desvio-padrão ~0.01
    W1, _, _, _ = learn.initialize_parameters(784, 128, 10)
    assert abs(W1.mean()) < 1e-3
    assert 0.009 < W1.std() < 0.011


def test_etapa1_relu_exemplo_do_pdf():
    a = learn.relu(np.array([[-1, 0, 1], [-2, 0, 2]]))
    assert np.array_equal(a, np.array([[0, 0, 1], [0, 0, 2]]))


def test_etapa1_relu_preserva_shape_e_nao_tem_negativos():
    Z = np.random.default_rng(0).normal(size=(5, 7))
    A = learn.relu(Z)
    assert A.shape == Z.shape
    assert np.all(A >= 0)
    assert np.array_equal(A[Z > 0], Z[Z > 0])   # positivos passam inalterados


def test_etapa1_softmax_exemplo_do_pdf():
    a = learn.softmax(np.array([[1, 2, 3], [4, 5, 6]]))
    assert np.allclose(a, np.vstack([PDF_SOFTMAX_ROW, PDF_SOFTMAX_ROW]), atol=ATOL)


def test_etapa1_softmax_linhas_somam_1_e_valores_entre_0_e_1():
    Z = np.random.default_rng(1).normal(size=(6, 10))
    S = learn.softmax(Z)
    assert S.shape == Z.shape
    assert np.allclose(S.sum(axis=1), 1.0)
    assert np.all((S >= 0) & (S <= 1))


def test_etapa1_softmax_opera_por_linha():
    # Cada linha deve ser normalizada de forma independente das demais
    Z = np.random.default_rng(2).normal(size=(4, 10))
    S = learn.softmax(Z)
    for i in range(Z.shape[0]):
        assert np.allclose(S[i], learn.softmax(Z[i:i + 1])[0])


def test_etapa1_softmax_estavel_para_valores_grandes():
    # Sem subtrair o máximo, exp(1000) = inf e o resultado vira NaN
    S = learn.softmax(np.array([[1000., 1001., 1002.], [-1000., 0., 1000.]]))
    assert np.all(np.isfinite(S))
    assert np.allclose(S[0], PDF_SOFTMAX_ROW, atol=ATOL)  # = softmax([1, 2, 3])
    assert np.allclose(S.sum(axis=1), 1.0)


def test_etapa1_softmax_invariante_a_deslocamento():
    # softmax(z - c) = softmax(z): fundamenta a subtração do máximo
    Z = np.random.default_rng(3).normal(size=(3, 10))
    assert np.allclose(learn.softmax(Z), learn.softmax(Z + 123.0))


def test_etapa1_onehotencoder_sparse_output():
    # Garante que o argumento usado em load_mnist existe na versão instalada
    # (sem baixar o MNIST)
    from sklearn.preprocessing import OneHotEncoder
    y = np.array([0, 2, 1]).reshape(-1, 1)
    out = OneHotEncoder(sparse_output=False).fit_transform(y)
    assert isinstance(out, np.ndarray)
    assert np.array_equal(out, np.eye(3)[[0, 2, 1]])


# ===========================================================================
# ETAPA 2 - forward
# ===========================================================================

def test_etapa2_forward_retorna_4_saidas_com_shapes_corretos():
    parameters, X = pdf_setup()
    out = learn.forward(X, parameters)
    assert isinstance(out, tuple) and len(out) == 4
    Z1, A1, Z2, A2 = out
    assert Z1.shape == (2, 4) and A1.shape == (2, 4)
    assert Z2.shape == (2, 10) and A2.shape == (2, 10)


def test_etapa2_forward_valores_do_pdf():
    parameters, X = pdf_setup()
    Z1, A1, Z2, A2 = learn.forward(X, parameters)
    assert np.allclose(Z1, PDF_Z1, atol=ATOL)
    assert np.allclose(A1, PDF_A1, atol=ATOL)
    assert np.allclose(Z2, PDF_Z2, atol=ATOL)
    assert np.allclose(A2, PDF_A2, atol=ATOL)


def test_etapa2_forward_A1_sem_negativos_e_A2_somando_1():
    parameters, X = pdf_setup()
    _, A1, _, A2 = learn.forward(X, parameters)
    assert np.all(A1 >= 0)
    assert np.allclose(A2.sum(axis=1), 1.0)


def test_etapa2_forward_ativacoes_coerentes_com_Z():
    # A1 = relu(Z1) e A2 = softmax(Z2) - a saída usa A1, não X
    rng = np.random.default_rng(4)
    params = random_parameters(rng, 6, 5, 3)
    Z1, A1, Z2, A2 = learn.forward(rng.normal(size=(4, 6)), params)
    assert np.allclose(A1, learn.relu(Z1))
    assert np.allclose(A2, learn.softmax(Z2))
    W2, b2 = params[2], params[3]
    assert np.allclose(Z2, A1 @ W2 + b2)


def test_etapa2_forward_igual_a_implementacao_ingenua():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(5, 7))
    params = random_parameters(rng, 7, 3, 4)
    for name, got, ref in zip(["Z1", "A1", "Z2", "A2"],
                              learn.forward(X, params),
                              naive_forward(X, params)):
        assert np.allclose(got, ref, atol=1e-12), name


def test_etapa2_forward_broadcasting_vies_para_qualquer_m():
    # O viés (1, H) deve ser somado a todas as linhas, para qualquer lote
    rng = np.random.default_rng(6)
    params = random_parameters(rng, 784, 8, 10)
    for m in (1, 2, 7, 1000):
        Z1, A1, Z2, A2 = learn.forward(rng.random((m, 784)), params)
        assert Z1.shape == (m, 8) and A2.shape == (m, 10)
        assert np.allclose(A2.sum(axis=1), 1.0)
    # Com X = 0, Z1 deve ser exatamente b1 repetido em cada linha
    Z1, *_ = learn.forward(np.zeros((3, 784)), params)
    assert np.allclose(Z1, np.repeat(params[1], 3, axis=0))


def test_etapa2_forward_ultimo_elemento_e_previsao_usada_pelo_server():
    # server.py faz learn.forward(X_test, parameters)[-1]
    parameters, X = pdf_setup()
    out = learn.forward(X, parameters)
    assert out[-1] is out[3]
    assert np.allclose(out[-1], PDF_A2, atol=ATOL)


def test_etapa2_forward_compativel_com_backward():
    """Gradient check: os gradientes do backward() (já pronto), alimentados com
    as saídas de forward(), devem bater com diferenças finitas da perda.
    A entropia cruzada aqui é local ao teste (a oficial é da Etapa 3)."""
    rng = np.random.default_rng(7)
    m, n, H, C = 5, 6, 4, 3
    X = rng.normal(size=(m, n))
    y = np.eye(C)[rng.integers(0, C, size=m)]
    params = [p.copy() for p in random_parameters(rng, n, H, C)]

    def loss(ps):
        A2 = learn.forward(X, tuple(ps))[-1]
        return -np.sum(y * np.log(A2)) / m

    Z1, A1, _, A2 = learn.forward(X, tuple(params))
    grads = learn.backward(X, y, params[2], Z1, A1, A2)

    eps = 1e-6
    for name, P, G in zip(["dW1", "db1", "dW2", "db2"], params, grads):
        assert G.shape == P.shape, name
        numeric = np.zeros_like(P)
        for idx in np.ndindex(P.shape):
            original = P[idx]
            P[idx] = original + eps
            plus = loss(params)
            P[idx] = original - eps
            minus = loss(params)
            P[idx] = original
            numeric[idx] = (plus - minus) / (2 * eps)
        rel = np.linalg.norm(G - numeric) / (np.linalg.norm(G) + np.linalg.norm(numeric))
        assert rel < 1e-7, f"{name}: erro relativo {rel:.1e}"


# ===========================================================================
# ETAPA 3 - cross_entropy_loss, accuracy
# ===========================================================================

# Seções 3.4 e 3.6 - mesmas matrizes para perda e acurácia
PDF_Y_TRUE = np.array([[0, 1, 0], [1, 0, 0]])
PDF_Y_PRED = np.array([[0.1, 0.6, 0.3], [0.3, 0.4, 0.3]])
PDF_LOSS = 0.8573992140459634
PDF_ACC = 0.5


def test_etapa3_loss_exemplo_do_pdf():
    loss = learn.cross_entropy_loss(PDF_Y_TRUE, PDF_Y_PRED)
    assert math.isclose(loss, PDF_LOSS, rel_tol=0, abs_tol=1e-15)


def test_etapa3_loss_retorna_float_python_serializavel():
    import json
    loss = learn.cross_entropy_loss(PDF_Y_TRUE, PDF_Y_PRED)
    assert type(loss) is float
    json.dumps({"loss": loss})  # como o server.py envia via SocketIO


def test_etapa3_loss_igual_a_formula_com_lacos():
    # L = -(1/m) * soma_i soma_c y_ic * ln(yhat_ic), com logaritmo natural
    rng = np.random.default_rng(8)
    y = np.eye(10)[rng.integers(0, 10, size=7)]
    p = learn.softmax(rng.normal(size=(7, 10)))
    ref = -sum(y[i, c] * math.log(p[i, c])
               for i in range(7) for c in range(10)) / 7
    assert math.isclose(learn.cross_entropy_loss(y, p), ref, rel_tol=1e-12)


def test_etapa3_loss_e_media_por_exemplo():
    # Duplicar os exemplos não pode mudar a perda (divide por m)
    y2 = np.vstack([PDF_Y_TRUE, PDF_Y_TRUE])
    p2 = np.vstack([PDF_Y_PRED, PDF_Y_PRED])
    assert math.isclose(learn.cross_entropy_loss(y2, p2), PDF_LOSS, rel_tol=1e-12)


def test_etapa3_loss_previsao_uniforme_vale_ln_C():
    # Saída uniforme 1/10 -> perda = ln(10) ≈ 2.302585 (perda inicial esperada)
    y = np.eye(10)[[3, 7, 0]]
    p = np.full((3, 10), 0.1)
    assert math.isclose(learn.cross_entropy_loss(y, p), math.log(10), rel_tol=1e-12)


def test_etapa3_loss_previsao_perfeita_vale_zero():
    y = np.eye(4)[[0, 2, 1]]
    assert abs(learn.cross_entropy_loss(y, y.astype(float))) < 1e-12


def test_etapa3_loss_finita_com_probabilidade_zero():
    # Classe correta com probabilidade 0.0 (underflow): clipping evita inf/NaN
    y = np.array([[1, 0, 0]])
    p = np.array([[0.0, 0.0, 1.0]])
    loss = learn.cross_entropy_loss(y, p)
    assert math.isfinite(loss)
    assert math.isclose(loss, -math.log(1e-15), rel_tol=1e-12)  # ≈ 34.54


def test_etapa3_loss_com_saida_do_forward():
    parameters, X = pdf_setup()
    A2 = learn.forward(X, parameters)[-1]
    y = np.eye(10)[[0, 1]]
    loss = learn.cross_entropy_loss(y, A2)
    assert type(loss) is float and abs(loss - math.log(10)) < 0.01


def test_etapa3_accuracy_exemplo_do_pdf():
    assert learn.accuracy(PDF_Y_TRUE, PDF_Y_PRED) == PDF_ACC


def test_etapa3_accuracy_retorna_float_python_serializavel():
    import json
    acc = learn.accuracy(PDF_Y_TRUE, PDF_Y_PRED)
    assert type(acc) is float
    json.dumps({"accuracy": acc})


def test_etapa3_accuracy_extremos_e_fracao():
    y = np.eye(3)[[0, 1, 2, 0]]
    assert learn.accuracy(y, y * 0.9 + 0.01) == 1.0          # tudo certo
    assert learn.accuracy(y, np.roll(y, 1, axis=1)) == 0.0   # tudo errado
    p = np.eye(3)[[0, 1, 0, 1]]                               # 2 de 4 certos
    assert learn.accuracy(y, p) == 0.5


def test_etapa3_accuracy_compara_por_linha():
    # Deve usar axis=1 (classe de cada exemplo), não o argmax global
    rng = np.random.default_rng(9)
    y = np.eye(10)[rng.integers(0, 10, size=50)]
    p = learn.softmax(rng.normal(size=(50, 10)))
    ref = sum(int(np.argmax(y[i]) == np.argmax(p[i])) for i in range(50)) / 50
    assert learn.accuracy(y, p) == ref


# ---------------------------------------------------------------------------
# Execução direta, sem pytest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [(name, fn) for name, fn in list(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  OK    {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FALHA {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} testes passaram")
    sys.exit(1 if failures else 0)
