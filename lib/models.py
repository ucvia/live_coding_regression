from cvxpy import Variable, Problem, Minimize, Parameter
from cvxpy import sum_squares, norm2, matmul, norm1


class OLS:
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y
        self.n, self.m = X.shape
        self.solver = None

        
        self.X_ols = Parameter(shape=self.X.shape, name="X")
        self.y_ols = Parameter(shape=self.y.shape, name="y")
        self.w_ols = Variable(shape=(self.m, 1), name="w_ols")

        self.X_ols.value, self.y_ols.value = self.X, self.y


    def resolver(self):
        ols = Problem(
            Minimize
            (
                sum_squares(self.y_ols - self.X_ols@self.w_ols)
            )
        )

        if self.solver is not None:
            ols.solve(solver=self.solver)
        else:
            ols.solve()

        return self.w_ols.value

class Lasso:
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y
        self.n, self.m = X.shape
        self.solver = None

        
        self.X_lasso = Parameter(shape=self.X.shape, name="X")
        self.y_lasso = Parameter(shape=self.y.shape, name="y")
        self.w_lasso = Variable(shape=(self.m, 1), name="w_lasso")

        self.X_lasso.value, self.y_lasso.value = self.X, self.y

    def resolver(self, _lambda=None, _s=None):
        if _s is None and _lambda is None:
            # Levantar exception con raise
            return "Error"
        
        if _s is not None:
            s_lasso = Parameter(nonneg=True, name="s")
            s_lasso.value = _s
            Lasso = Problem(
                Minimize(
                    norm2(self.y_lasso - self.X_lasso@self.w_lasso)**2
                ),
                [
                    norm1(self.w_lasso) <= s_lasso
                ]
            )

        if _lambda is not None:
            l_lasso = Parameter(nonneg=True, name="lambda")
            l_lasso.value = _lambda
            Lasso = Problem(
                Minimize(
                    norm2(self.y_lasso - self.X_lasso@self.w_lasso)**2 + l_lasso * norm1(self.w_lasso) 
                )
            )

        if self.solver is not None:
            Lasso.solve(solver=self.solver)
        else:
            Lasso.solve()

        return self.w_lasso.value
   