from fractions import Fraction


class MetaMatrix(type):
    def __str__(cls):
        return f"<class \'{cls.__name__}'>"


class Matrix(metaclass=MetaMatrix):
    def __init__(self, matrix):
        self.__name__ = "matrix"

        type(self).__name__ = "matrix"
        isComplex = type(matrix).__name__ == "complex"
        if isComplex:
            matrix = [[matrix.real, -1 * matrix.imag], [matrix.imag, matrix.real]]
        self.matrix = matrix
        self.order = self.orderInit(self.matrix)
        self.identity = self.isIdentity()

    def __str__(self):
        matrixOutput = "\n"
        matrixRow = []
        matrixRow += "["
        rowFormat = "{:^10}" * self.order[1]
        topRowFormat = "{:<0}" + rowFormat
        bottomRowFormat = rowFormat + "{:>0}"

        for r in range(self.order[0]):
            for c in range(self.order[1]):
                matrixRow.append(str(self.matrix[r][c]))

            if r == 0:
                matrixOutput += topRowFormat.format(*matrixRow)
            elif r == self.order[0] - 1:
                matrixRow += "]"
                matrixOutput += bottomRowFormat.format(*matrixRow)
            else:
                matrixOutput += rowFormat.format(*matrixRow)

            matrixRow = []
            if r != self.order[0] - 1:
                matrixOutput += "\n"
                matrixOutput += " "
        matrixOutput += "\n"
        return matrixOutput

    def __complex__(self):
        return complex(self[0][0], self[1][0])

    def __getitem__(self, *args):
        i = args[0]
        return self.matrix[i]

    def __setitem__(self, *args):
        i = args[0]
        val = args[1]
        self.matrix[i] = val

    def __eq__(self, other):
        try:
            return self.__name__ == other.__name__
        except AttributeError:
            return False

    def __add__(self, matrix2):
        if self.__name__ != matrix2.__name__:
            raise Exception(f"TypeError: unsupported operand type(s) for +: \'matrix' and \'{type(matrix2).__name__}'")
        if self.order != matrix2.order:
            raise Exception(f"OrderError: matrices must have the same order")

        addingMatrix = []
        for r in range(len(self.matrix)):
            addingMatrix.append([])
            for c in range(len(self.matrix[r])):
                addingMatrix[r].append(self.matrix[r][c] + matrix2.matrix[r][c])
        return Matrix(addingMatrix)

    def __sub__(self, matrix2):
        if self.__name__ != matrix2.__name__:
            raise Exception(f"TypeError: unsupported operand type(s) for -: \'matrix' and \'{type(matrix2).__name__}'")

        subtractingMatrix = matrix2 * -1
        return self.__add__(subtractingMatrix)

    def __mul__(self, obj2):
        # scalar multiplication
        if type(obj2).__name__ == "int" or type(obj2).__name__ == "float":
            multiplyingMatrix = []
            for r in range(len(self.matrix)):
                multiplyingMatrix.append([])
                for c in range(len(self.matrix[r])):
                    multiplyingMatrix[r].append(self.matrix[r][c] * obj2)
            return self.roundMatrix(Matrix(multiplyingMatrix))

        # matrix multiplication
        if type(obj2).__name__ == "matrix":
            if self.order[1] != obj2.order[0]:
                raise Exception(f"OrderError: number of columns in first matrix must equal that of rows in the second")

            multiplyingMatrix = []
            dotProductVal = 0

            for i in range(self.order[0]):
                multiplyingMatrix.append([])
                # col
                for j in range(obj2.order[1]):
                    # col
                    for k in range(self.order[1]):
                        dotProductVal += (self.matrix[i][k] * obj2.matrix[k][j])

                    multiplyingMatrix[i].append(dotProductVal)
                    dotProductVal = 0
            return self.roundMatrix(Matrix(multiplyingMatrix))

    def __rmul__(self, obj2):
        if type(obj2).__name__ == "int" or type(obj2).__name__ == "float":
            return self.__mul__(obj2)
        else:
            return obj2.__mul__(self)

    def __truediv__(self, obj2):
        if type(obj2).__name__ == "int" or type(obj2).__name__ == "float":
            return self * (obj2 ** -1)

        if type(obj2).__name__ == "matrix":
            print(obj2.inverse())
            return self * obj2.inverse()

    def __rtruediv__(self, obj2):
        if type(obj2).__name__ == "int" or type(obj2).__name__ == "float":
            return self.__truediv__(obj2)
        else:
            return obj2.__truediv__(self)

    @staticmethod
    def orderInit(matrix):
        colNum = len(matrix[0])
        for r in range(1, len(matrix)):
            if colNum != len(matrix[r]):
                raise Exception(f"SyntaxError: matrix must not have rows of different sizes")
            colNum = len(matrix[r])

        try:
            # (m x n)
            return tuple((len(matrix), len(matrix[0])))
        except TypeError:
            return tuple((len(matrix), 1))

    def det(self, matrix):
        determinant = 0
        subOrder = self.orderInit(matrix)
        if self.order[0] != self.order[1]:
            raise Exception("OrderError: matrix must be square to find the determinant")
        if subOrder[0] == 2:
            determinant += self.det2x2(matrix)
        else:
            # laplace expansion column-wise
            for r in range(subOrder[1]):
                sub = self.subMatrix(matrix, r, 0)
                test = ((-1) ** r) * matrix[r][0] * self.det(sub)
                determinant += test

        return determinant

    @staticmethod
    def subMatrix(matrix, row, col):
        sub = []
        rowSkipped = False
        for r in range(len(matrix)):
            if r == row:
                rowSkipped = True
            else:
                sub.append([])
                for c in range(0, len(matrix[r])):
                    if c != col:
                        if rowSkipped:
                            sub[r - 1].append(matrix[r][c])
                        else:
                            sub[r].append(matrix[r][c])

        return sub

    @staticmethod
    def det2x2(sub):
        return sub[0][0] * sub[1][1] - sub[0][1] * sub[1][0]

    def cofactor(self):
        cofactorMatrix = []
        for r in range(self.order[0]):
            cofactorMatrix.append([])
            for c in range(self.order[1]):
                cofactor = (-1) ** (c + r) * self.det(self.subMatrix(self.matrix, r, c))
                cofactorMatrix[r].append(cofactor)

        return Matrix(cofactorMatrix)

    def transpose(self, matrix=None):
        if matrix is None:
            matrix = self.matrix
        transposedMatrix = []
        for c in range(self.order[1]):
            transposedMatrix.append([])
            for r in range(self.order[0]):
                transposedMatrix[c].append(matrix[r][c])
        return Matrix(transposedMatrix)

    def adjoint(self):
        return self.transpose(self.cofactor())

    def inverse(self):
        determinant = self.det(self.matrix)
        if determinant == 0:
            raise Exception("InversionError: matrix is invertible when det is 0")
        return self.adjoint() / determinant

    def isIdentity(self):
        if self.order[0] != self.order[1]:
            return False
        for r in range(self.order[0]):
            if self.matrix[r][r] == 1 or self.matrix[r][r] == 1:
                if any(self.matrix[r][:r] + self.matrix[r][r + 1:]):
                    return False
        return True

    def roundMatrix(self, matrix=None):
        isObject = False
        if matrix is None:
            matrix = self.matrix
            isObject = True
        for r in range(matrix.order[0]):
            for c in range(matrix.order[1]):
                if type(matrix[r][c]).__name__ == "float":
                    matrix[r][c] = Fraction(matrix[r][c]).limit_denominator(10000)
                else:
                    matrix[r][c] = int(matrix[r][c])
                if matrix[r][c] == -0.0:
                    matrix[r][c] = 0

        if not isObject:
            return matrix


class FormattedOrder:
    def __init__(self, order):
        self.order = order

    def __str__(self):
        return f"{self.order[0]} x {self.order[1]}"


array1 = [[317, 22, 909], [0, 1, 0], [12, 38, 3]]
array2 = [[7, 8, 9], [9, 10, 11], [11, 11, 12]]

matrix1 = Matrix(array1)
matrix2 = Matrix(array2)

print(matrix1.inverse())
