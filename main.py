# made by charmanic :)

from fractions import Fraction


class FormattedOrder:
    """Format order in the form m x n when outputted, instead of as a tuple."""
    def __init__(self, order):
        self.order = order

    def __str__(self):
        return f"{self.order[0]} x {self.order[1]}"


class MetaMatrix(type):
    """Ensure file type is displayed correctly."""
    def __str__(cls):
        return f"<class \'{cls.__name__}'>"


class Matrix(metaclass=MetaMatrix):
    """Create and return a new object."""
    def __init__(self, matrix):
        # Ensure file type name is matrix.
        type(self).__name__ = "matrix"
        
        # Check if inputted value is a complex number and convert it into a 2x2 matrix.
        isComplex = type(matrix).__name__ == "complex"
        if isComplex:
            # Where a complex number is a + ib, its corresponding matrix will be:
            # [ a -b
            #   b  a ] 
            matrix = [[matrix.real, -1 * matrix.imag], [matrix.imag, matrix.real]]
        
        self.matrix = self.checkEntries(matrix)
        self.order = self.orderInit(self.matrix)
        self.identity = self.isIdentity()

    def __add__(self, other):
        """Handle matrix addition."""
        # Check if operands are of 'matrix' data type and if of equivalent orders.
        if self.__name__ != other.__name__:
            raise TypeError(f"unsupported operand type(s) for +: \'matrix' and \'{type(other).__name__}'")
        if self.order != other.order:
            raise Exception(f"OrderError: matrices must have the same order")

        # Add each entry in left operand to the corresponding entry in the right operand, i.e.:
        # [ 1  3      +    [ 2  4      =    [ 1 + 2  3 + 4
        #   5  7 ]           6  8 ]           5 + 6  7 + 8 ]
        
        addingMatrix = []
        for r in range(len(self.matrix)):
            addingMatrix.append([])
            for c in range(len(self.matrix[r])):
                addingMatrix[r].append(self.matrix[r][c] + other.matrix[r][c])
        
        return Matrix(addingMatrix)

    def __complex__(self):
        """Convert appropriate 2x2 matrices into complex numbers."""
        if self.order != (2, 2):
            raise Exception(f"OrderError: conversion to complex number requires order of (2, 2)")
        if self.matrix[0][0] == self.matrix[1][1] and -1 * self.matrix[0][1] == self.matrix[1][0]:
            return complex(self[0][0], self[1][0])
        
        raise TypeError(f"unable to convert matrix into complex number")

    def __eq__(self, other):
        """Handle equality checking"""
        try:
            return self.__name__ == other.__name__
        except AttributeError:
            return False

    def __getitem__(self, *args):
        """Handle entry accessing."""
        i = args[0]
        return self.matrix[i]

    def __setitem__(self, *args):
        """Handle entry value."""
        i = args[0]
        val = args[1]
        self.matrix[i] = val

    def __mul__(self, other):
        """Handle scalar multiplication and matrix multiplication."""

        # Calculate scalar multiplication if other is an int or float.
        
        # Multiply each entry in the matrix by the scale, i.e.:
        # 2    *    [ 1  2      =    [ 1 * 2  2 * 2
        #             3  4 ]           3 * 2  4 * 2 ]
        
        if type(other).__name__ == "int" or type(other).__name__ == "float":
            multiplyingMatrix = []
            for r in range(len(self.matrix)):
                multiplyingMatrix.append([])
                for c in range(len(self.matrix[r])):
                    multiplyingMatrix[r].append(self.matrix[r][c] * other)
            return self.roundMatrix(Matrix(multiplyingMatrix))

        # Calculate matrix multiplication if other is a matrix.

        # Multiply corresponding entries of the matrices together and sum their results, i.e.:
        # [ 1  2      *    [ 5  6      =    [ 1 * 5 + 2 * 7   1 * 6 + 2 * 8
        #   3  4 ]           7  8 ]           3 * 5 + 4 * 7   3 * 6 + 4 * 8 ]
        
        if type(other).__name__ == "matrix":
            # Check if number of columns in left operand equals that of rows in right operand.
            if self.order[1] != other.order[0]:
                raise Exception(f"OrderError: number of columns in first matrix must equal that of rows in the second")

            multiplyingMatrix = []
            productVal = 0

            # Iterate through rows of left matrix.
            for i in range(self.order[0]):
                multiplyingMatrix.append([])
                # Iterate through columns of right matrix.
                for j in range(other.order[1]):
                    # Iterate through columns of left matrix.
                    for k in range(self.order[1]):
                        productVal += (self.matrix[i][k] * other.matrix[k][j])
                    # Add value of product to current row in multiplyingMatrix
                    multiplyingMatrix[i].append(productVal)
                    productVal = 0
            
            return self.roundMatrix(Matrix(multiplyingMatrix))

    def __rmul__(self, other):
        """Handle multiplication if object is the right operand."""
        if type(other).__name__ == "int" or type(other).__name__ == "float":
            return self.__mul__(other)
        else:
            return other.__mul__(self)
    
    def __str__(self):
        """Ensure matrix is displayed appropriately.""" 
        # A matrix from the array [[1, 2, 3], [4, 5, 6]] would be displayed as such:
        # [ 1 2 3
        #  4 5 6 ] 
        matrixOutput = "\n"
        matrixRow = []
        matrixRow += "["
        
        # Set up format strings to allow for adequate space between entries.
        rowFormat = "{:^10}" * self.order[1]
        topRowFormat = "{:<0}" + rowFormat
        bottomRowFormat = rowFormat + "{:>0}"

        # Append entries to matrixRow and format them appropriately.
        for r in range(self.order[0]):
            for c in range(self.order[1]):
                matrixRow.append(str(self.matrix[r][c]))

            # First line formatting.
            if r == 0:
                matrixOutput += topRowFormat.format(*matrixRow)
            # Last line formatting.
            elif r == self.order[0] - 1:
                matrixRow += "]"
                matrixOutput += bottomRowFormat.format(*matrixRow)
            # Typical line formatting.
            else:
                matrixOutput += rowFormat.format(*matrixRow)

            matrixRow = []
            # Add a new line and a space after every row.
            if r != self.order[0] - 1:
                matrixOutput += "\n"
                matrixOutput += " "
        matrixOutput += "\n"
        return matrixOutput

    def __sub__(self, other):
        """Handle matrix subtraction."""
        # Check if operands are of 'matrix' data type.
        if self.__name__ != other.__name__:
            raise TypeError(f"unsupported operand type(s) for -: \'matrix' and \'{type(other).__name__}'")

        # A - B = A + -(1 * B)
        subtractingMatrix = other * -1
        return self.__add__(subtractingMatrix)

    def __truediv__(self, other):
        """Handle scalar and matrix division."""

        # A / n = A * ( 1 / n)
        if type(other).__name__ == "int" or type(other).__name__ == "float":
            return self * (other ** -1)
    
        # A / B = A * B ^ -1 (if |B| is non-zero)     
        if type(other).__name__ == "matrix":
            print(other.inverse())
            return self * other.inverse()

    def __rtruediv__(self, other):
        """Handle division if object is the right operand."""
        if type(other).__name__ == "int" or type(other).__name__ == "float":
            return self.__truediv__(other)
        else:
            return other.__truediv__(self)
    
    @staticmethod
    def checkEntries(matrix):
        """Check for presence of types other than int and float in entries, raising an exception if applicable."""
        for r in matrix:
            for c in matrix:
                if type(c).__name__ != "int" or type(c).__name__ != "float":
                    raise TypeError(f"entries cannot be of type {type(c)}")

    @staticmethod
    def det2x2(sub):
        # Calculate determinant of 2x2 matrix, i.e.:
        # | [ 1  2   |    =    1 * 4 - 2 * 3 = -2
        # |   3  4 ] |
        return sub[0][0] * sub[1][1] - sub[0][1] * sub[1][0]
    
    @staticmethod
    def orderInit(matrix):
        """Return the order of the matrix in tuple form (m, n) where m is the number of rows and n that of columns."""
        colNum = len(matrix[0])
        for r in range(1, len(matrix)):
            # Raise exception for matrices with different row lengths - e.g.:
            # [ 3 2
            #   1 2 3 ]
            if colNum != len(matrix[r]):
                raise SyntaxError(f"matrix must not have rows of different sizes")
            colNum = len(matrix[r])

        # Return order as tuple.
        try:
            return tuple((len(matrix), len(matrix[0])))
        # Except vector matrices.
        except TypeError:
            return tuple((len(matrix), 1))

    @staticmethod
    def subMatrix(matrix, row, col):
        """Deduce the submatrix of the object formed by remove the current row and column of entry."""
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

    def adjoint(self):
        """Determine the adjugate matrix of the object."""
        # adj(A) = C^T
        # Such that:
        # adj(A) = adjoint of A, C^T = transpose of cofactor matrix.
        
        return self.transpose(self.cofactor())

    def cofactor(self):
        """Calculate comatrix of matrix."""
        # C(ij) = -1 ^ (i + j) * M(i,j)
        # Such that: 
        # C(ij) = cofactor at index (i,j), M(i,j) = minor excluding row i and column j. 
        
        comatrix = []
        for r in range(self.order[0]):
            # Iterate through rows of comatrix
            comatrix.append([])
            for c in range(self.order[1]):
                # Calculate cofactor at (i, j).
                cofactor = (-1) ** (c + r) * self.det(self.subMatrix(self.matrix, r, c))
                comatrix[r].append(cofactor)

        return Matrix(comatrix)
    
    def det(self, matrix):
        """Calculate the determinant of matrix (whereby its order is n x n || n >= 2) via the Laplace expansion."""
        determinant = 0
        subOrder = self.orderInit(matrix)
        # Check if matrix is square.
        if self.order[0] != self.order[1]:
            raise Exception("OrderError: matrix must be square to find the determinant")
        # Resolve 2x2 matrix case.
        if subOrder[0] == 2:
            determinant += self.det2x2(matrix)
        else:
            # Laplace expansion along column j:
            #       n
            # |A| = Σ ( (-1^(i + j)) * A(ij) * m(ij)  )
            #      i=1   
            # Such that:
            # |A| = determinant of A, n x n = order of A, 
            # A(ij) = entry at (i, j) of Adj(A), m(ij) = minor formed by exclusion of row i and column j.
            # (It is inefficient but I'm not going for efficiency here.)
            
            for r in range(subOrder[1]):
                sub = self.subMatrix(matrix, r, 0)
                rowValue = ((-1) ** r) * matrix[r][0] * self.det(sub)
                determinant += rowValue

        return determinant
    
    def inverse(self):
        """Determine the inverse of the matrix."""
        # A ^ -1 = Adj(A) / |A|
        # Such that:
        # A ^ -1 = inverse of A, Adj(A) = adjoint of A, |A| = determinant of A.
        
        determinant = self.det(self.matrix)
        # Division by zero, so where |A| = 0, A ^ -1 is undefined (A is singular).
        if determinant == 0:
            raise Exception("InversionError: matrix is invertible when det is 0")
        return self.adjoint() / determinant

    def isIdentity(self):
        """Check to see if the object is the identity matrix."""
        # I2 =  [ 1  0 
        #         0  1 ]
        
        # (A * A ^ -1 = I)
        if self.order[0] != self.order[1]:
            return False
        # Check to see if ones are on the main diagonal and if there are zeroes everywhere else.
        for r in range(self.order[0]):
            if self.matrix[r][r] == 1 or self.matrix[r][r] == 1:
                if any(self.matrix[r][:r] + self.matrix[r][r + 1:]):
                    return False
        return True

    def roundMatrix(self, matrix=None):
        """Convert floats in the matrix into fractions using the Fractions data type."""
        isObject = False
        if matrix is None:
            matrix = self.matrix
            isObject = True
        
        for r in range(matrix.order[0]):
            for c in range(matrix.order[1]):
                # Check to see if entry is a float.
                if type(matrix[r][c]).__name__ == "float":
                    matrix[r][c] = Fraction(matrix[r][c]).limit_denominator(10000)
                else:
                    matrix[r][c] = int(matrix[r][c])
                # Resolve any instances of -0.0
                if matrix[r][c] == -0.0:
                    matrix[r][c] = 0

        if not isObject:
            return matrix
            
    def transpose(self, matrix=None):
        """Determine transpose of matrix."""
        # Swap the rows and columns of the matrix, i.e.:
        # [ 1  2  ^ T    =    [ 1  3
        #   3  4 ]              2  4 ]
        
        if matrix is None:
            matrix = self.matrix
        transposedMatrix = []
        for c in range(self.order[1]):
            transposedMatrix.append([])
            for r in range(self.order[0]):
                transposedMatrix[c].append(matrix[r][c])
        return Matrix(transposedMatrix)
