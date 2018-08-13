from platypus import (AdaptiveTimeContinuation, EpsilonProgressContinuation,
                      RandomGenerator, TournamentSelector,
                      NSGAII, EpsNSGAII, EpsilonBoxArchive, Multimethod,
                      GAOperator, SBX, PM, PCX,
                      DifferentialEvolution, UNDX, SPX, UM)

from ema_workbench.em_framework.optimization import AbstractConvergenceMetric

from math import sqrt


class NSGAIIHybrid(EpsilonProgressContinuation):
    def __init__(self,
                 problem,
                 epsilons,
                 population_size=100,
                 generator=RandomGenerator(),
                 selector=TournamentSelector(2),
                 variator=None,
                 **kwargs):

        L = len(problem.parameters)

        # -------------------------------------------------------------------
        #                           DefaultValue            BorgValue
        # PM   probability          1.0                     1.0 / L
        #      distribution index   20                      < 100 (20)
        #      source     Borg: An Auto-Adaptive MOEA Framework - Hadka, Reed
        #
        # SBX  probability          1.0                     > 0.8 (1.0)
        #      distribution index   15                      < 100 (15)
        #      source     Borg: An Auto-Adaptive MOEA Framework - Hadka, Reed;
        #                 Simulated Binary Crossover for Continuous Search
        #                 Space - Deb, Agrawal
        #
        # PCX  nparents             10                      3 (10)
        #      noffspring           2                       2-15 (2)
        #      eta                  0.1                     (0.1)
        #      zeta                 0.1                     (0.1)
        #      source     A Computationally Efficient Evolutionary Algorithm
        #                 for Real-Parameter Optimization - Deb et al 2002
        #
        # DE   crossover rate       0.1                     0.6 (0.1)
        #      step size            0.5                     0.6 (0.5)
        #      source     Borg: An Auto-Adaptive MOEA Framework - Hadka, Reed
        #
        # UNDX nparents             10                      3 (10)
        #      noffspring           2                       2 (2)
        #      zeta                 0.5                     0.5
        #      eta                  0.35                    0.35/sqrt(L) (0.35)
        #      source     Borg: An Auto-Adaptive MOEA Framework - Hadka, Reed;
        #                 A Computationally Efficient Evolutionary Algorithm
        #                 for Real-Parameter Optimization - Deb et al 2002
        #
        # SPX  nparents             10                      L + 1 (10)
        #      noffspring           2                       L + 1 (2)
        #      expansion            None                    sqrt((L+1)+1) (3.0)
        #      source     Borg: An Auto-Adaptive MOEA Framework - Hadka, Reed;
        #                 Multi-parent Recombination with Simplex Crossover
        #                 in Real Coded Genetic Algorithms - Tsutsui
        #
        # UM   probability          1                       1.0 / L
        #      source     Borg: An Auto-Adaptive MOEA Framework - Hadka, Reed
        # -------------------------------------------------------------------

        variators = [GAOperator(SBX(probability=1.0,
                                    distribution_index=15.0),
                                PM(probability=1.0/L,
                                   distribution_index=20.0)),
                     GAOperator(PCX(nparents=3, noffspring=2,
                                    eta=0.1, zeta=0.1),
                                PM(probability=1.0/L,
                                   distribution_index=20.0)),
                     GAOperator(DifferentialEvolution(crossover_rate=0.6,
                                                      step_size=0.6),
                                PM(probability=1.0/L,
                                   distribution_index=20.0)),
                     GAOperator(UNDX(nparents=3, noffspring=2,
                                     zeta=0.5, eta=0.35/sqrt(L)),
                                PM(probability=1.0/L,
                                   distribution_index=20.0)),
                     GAOperator(SPX(nparents=L+1, noffspring=L+1,
                                    expansion=sqrt(L+2)),
                                PM(probability=1.0/L,
                                   distribution_index=20.0)),
                     UM(probability=1/L)]

        variator = Multimethod(self, variators)

        super(NSGAIIHybrid, self).__init__(
                NSGAII(problem,
                       population_size,
                       generator,
                       selector,
                       variator,
                       EpsilonBoxArchive(epsilons),
                       **kwargs))


class OperatorProbabilities(AbstractConvergenceMetric):

    def __init__(self, name, index):
        super(OperatorProbabilities, self).__init__(name)
        self.index = index

    def __call__(self, optimizer):
        try:
            props = optimizer.algorithm.variator.probabilities
            self.results.append(props[self.index])
        except AttributeError:
            pass
