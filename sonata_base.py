import numpy as np
import quaternion as qt # type: ignore


def quaternion_to_complex(q, splitting="symp"):
    """Splits a quaternion array into a set of two complex arrays.

    Two splittings are available:

    - Symplectic ('symp') (default)::

    q = z_1+iz_2

    where z_1, z_2 are complex in C_j encoded as (1, 1j) numpy arrays.

    - Cayley-Dickson (cd)::

    q = z_1+z_2j

    where z_1, z_2 are complex in C_i encoded as (1, 1j) numpy arrays.

    Args:
        q (quaternion or complex or real): quaternion numpy array
        splitting (str, optional): Splitting parameter: Cayley-Dickson 'cd' or symplectic 'symp'. Defaults to 'symp'.

    Returns:
        z_1, z_2 (complex): two complex numpy arrays.

    See also:
    complex_to_quaternion
    """

    q = np.array(q)

    if splitting.lower() == "symp":
        if q.dtype == "quaternion":
            array_q = qt.as_float_array(q)
            z_1 = array_q[..., 0] + array_q[..., 2] * 1j
            z_2 = array_q[..., 1] + array_q[..., 3] * 1j
        else:
            z_1 = q
            z_2 = np.zeros_like(z_1)

    elif splitting.lower() == "cd":
        if q.dtype == "quaternion":
            array_q = qt.as_float_array(q)
            z_1 = array_q[..., 0] + array_q[..., 1] * 1j
            z_2 = array_q[..., 2] + array_q[..., 3] * 1j
        else:
            z_1 = q
            z_2 = np.zeros_like(z_1)

    else:
        raise ValueError(
            "Splitting must be either 'cd' for Cayley-Dickson or 'symp' for symplectic."
        )

    return z_1, z_2


def complex_to_quaternion(z_1, z_2, splitting="symp"):
    """Constructs a quaternion array from two complex arrays.

    Two constructions are available:

    - Symplectic ('symp') (default)::

    q = z_1+iz_2

    where z_1, z_2 are encoded as (1, 1j) numpy arrays but are in fact in C_j.

    - Cayley-Dickson ('cd')::

    q = z_1+z_2j

    where z_1, z_2 are encoded as (1, 1j) numpy arrays but are in fact in C_i.

    Args:
        z_1, z_2 (complex or real): two complex numpy arrays.
        splitting (str, optional): Splitting parameter: Cayley-Dickson ('cd') or symplectic ('symp'). Defaults to 'symp'.

    Returns:
        q (quaternion): quaternion numpy array.

    See also:
    quaternion_to_complex
    """
    z_1 = np.array(z_1)
    z_2 = np.array(z_2)

    if list(z_1.shape) != list(z_2.shape):
        raise ValueError("Array 'z1' and 'z2' should be of same shape.")

    array_dimension = list(z_1.shape)
    array_dimension.append(4)
    array_q = np.zeros(tuple(array_dimension))

    if splitting.lower() == "symp":
        array_q[..., 0] = np.real(z_1)
        array_q[..., 1] = np.real(z_2)
        array_q[..., 2] = np.imag(z_1)
        array_q[..., 3] = np.imag(z_2)

    elif splitting.lower() == "cd":
        array_q[..., 0] = np.real(z_1)
        array_q[..., 1] = np.imag(z_1)
        array_q[..., 2] = np.real(z_2)
        array_q[..., 3] = np.imag(z_2)

    else:
        raise ValueError(
            "Splitting must be either 'cd' for Cayley-Dickson or 'symp' for symplectic."
        )

    return qt.as_quat_array(array_q)


def quaternion_to_adjoint(Q, splitting="symp"):
    """Builds the complex adjoint matrix of a quaternion array.

    Let Q a quaternion matrix of size IxJ.
    If (Z_1,Z_2) denotes the splitting arrays of the matrix Q, then the complex adjoint matrix chiQ is defined as:

    - For the Cayley-Dickson splitting (Q = Z_1+Z_2j with Z_1 and Z_2 in C_i)::

    chiQ = |  Z_1  Z_2  |.
           | -Z_2* Z_1* |

    - For the symplectic splitting (Q = Z_1+iZ_2 with Z_1 and Z_2 in C_j)::

    chiQ = | Z_1 -Z_2* |.
           | Z_2  Z_1* |

    Args:
        Q (quaternion or complex or real): quaternion numpy array of size IxJ.
        splitting (str, optional): Splitting parameter: Cayley-Dickson ('cd') or symplectic ('symp'). Defaults to 'symp'.

    Returns:
        chiQ (complex): complex numpy array of size (2I)x(2J).

    See also:
    quaternion_to_complex
    """

    Q = np.array(Q)

    array_dimension = np.shape(Q)

    if len(array_dimension) > 2:
        raise ValueError("Array 'Q' must be either a matrix or a vector.")

    if len(array_dimension) == 2:
        NUMBER_ROWS, NUMBER_COLUMNS = np.shape(Q)

    if len(array_dimension) == 1:
        Q = Q.reshape(-1, 1)
        NUMBER_ROWS, NUMBER_COLUMNS = np.shape(Q)

    if len(array_dimension) == 0:
        Q = Q.reshape(1, 1)
        NUMBER_ROWS, NUMBER_COLUMNS = np.shape(Q)

    chi_Q = np.zeros((2 * NUMBER_ROWS, 2 * NUMBER_COLUMNS), dtype=complex)

    Z_1, Z_2 = quaternion_to_complex(Q, splitting=splitting)

    if splitting.lower() == "symp":
        chi_Q[:NUMBER_ROWS, :NUMBER_COLUMNS] = Z_1
        chi_Q[:NUMBER_ROWS, NUMBER_COLUMNS:] = -np.conjugate(Z_2)
        chi_Q[NUMBER_ROWS:, :NUMBER_COLUMNS] = Z_2
        chi_Q[NUMBER_ROWS:, NUMBER_COLUMNS:] = np.conjugate(Z_1)

    elif splitting.lower() == "cd":
        chi_Q[:NUMBER_ROWS, :NUMBER_COLUMNS] = Z_1
        chi_Q[:NUMBER_ROWS, NUMBER_COLUMNS:] = Z_2
        chi_Q[NUMBER_ROWS:, :NUMBER_COLUMNS] = -np.conjugate(Z_2)
        chi_Q[NUMBER_ROWS:, NUMBER_COLUMNS:] = np.conjugate(Z_1)

    else:
        raise ValueError(
            "Splitting must be either 'cd' for Cayley-Dickson or 'symp' for symplectic."
        )

    return chi_Q


def adjoint_to_quaternion(chi_Q, splitting="symp"):
    """Builds the quaternion array of a complex adjoint matrix.

    Args:
        chi_Q (complex or real): complex numpy array of size (2I)x(2J).
        splitting (str, optional): Splitting parameter: Cayley-Dickson ('cd') or symplectic ('symp'). Defaults to 'symp'.

    Returns:
        Q (quaternion): quaternion array of size IxJ.

    See also:
    quaternion_to_adjoint
    complex_to_quaternion
    """
    array_dimension = np.shape(chi_Q)

    if len(array_dimension) != 2:
        raise ValueError("The array 'chi_Q' must be a matrix.")

    if any(np.mod(array_dimension, 2) > 0):
        raise ValueError("The dimensions of 'chi_Q' must be multiples of 2.")

    NUMBER_ROWS = int(array_dimension[0] / 2)
    NUMBER_COLUMNS = int(array_dimension[1] / 2)

    if splitting.lower() == "symp":
        Z_1 = chi_Q[:NUMBER_ROWS, :NUMBER_COLUMNS]
        Z_2 = chi_Q[NUMBER_ROWS:, :NUMBER_COLUMNS]

    elif splitting.lower() == "cd":
        Z_1 = chi_Q[:NUMBER_ROWS, :NUMBER_COLUMNS]
        Z_2 = chi_Q[:NUMBER_ROWS, NUMBER_COLUMNS:]

    else:
        raise ValueError(
            "Splitting must be either 'cd' for Cayley-Dickson or 'symp' for symplectic."
        )

    return complex_to_quaternion(Z_1, Z_2, splitting=splitting)


def ldot(A, B, splitting="symp"):
    """Computes the left-matrix product between two quaternionic arrays.

    For two quaternion-valued matrix A = (a_{ik}) and B = (b_{kj}), the left-matrix product is the matrix AB = (ab_{ij}) of size IxJ defined by ab_{ij} = sum_{k=1}^K a_{ik}b_{kj}.

    This function handles every combination of cases where A or B can be real, complex or quaternion matrices. For the complex case, the splitting is very important to mention, as the Cayley-Dickson splitting gives complex arrays in C_i whereas the symplectic gives complex arrays in C_j.

    Args:
        A (quaternion or complex or real): quaternion array of size IxK,
        B (quaternion): quaternion array of size KxJ,
        splitting (str, optional): Splitting parameter: Cayley-Dickson ('cd') or symplectic ('symp'). Defaults to 'symp'.

    Returns:
        AB (quaternion): quaternion array of size IxJ.
    """

    A_1, A_2 = quaternion_to_complex(A, splitting=splitting)
    B_1, B_2 = quaternion_to_complex(B, splitting=splitting)

    if splitting.lower() == "symp":
        return complex_to_quaternion(
            np.dot(A_1, B_1) - np.dot(np.conjugate(A_2), B_2),
            np.dot(np.conjugate(A_1), B_2) + np.dot(A_2, B_1),
            splitting=splitting,
        )
    elif splitting.lower() == "cd":
        return complex_to_quaternion(
            np.dot(A_1, B_1) - np.dot(A_2, np.conjugate(B_2)),
            np.dot(A_1, B_2) + np.dot(A_2, np.conjugate(B_1)),
            splitting=splitting,
        )
    else:
        # Double check. This error should have be raised when 'quaternion_to_complex' is ran.
        raise ValueError(
            "Splitting must be either 'cd' for Cayley-Dickson or 'symp' for symplectic."
        )


def rdot(A, B, splitting="symp"):
    """Computes the right-matrix product of two quaternion arrays.

    For two quaternion-valued matrix A = (a_{ik}) and B = (b_{kj}), the right-matrix product is the matrix AB = (ab_{ij}) of size IxJ defined by ab_{ij} = sum_{k=1}^K b_{kj}a_{ik}.

    It holds that rdot(A,B) = (ldot(B^T,A^T))^T.

    Args:
        A (quaternion or complex or real): array of size IxK,
        B (quaternion or complex or real): array of size KxJ,
        splitting (str, optional): Splitting parameter: Cayley-Dickson ('cd') or symplectic ('symp'). Defaults to 'symp'.

    Returns:
        AB (quaternion): quaternion array of size IxJ.

    See also:
    ldot
    """

    return np.transpose(ldot(np.transpose(B), np.transpose(A), splitting=splitting))

def round(q):
    q1, q2 = quaternion_to_complex(q)
    return complex_to_quaternion(np.round(q1),np.round(q2))
