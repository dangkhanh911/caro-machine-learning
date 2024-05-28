from caro import Caro, Cell, Result
from rlagent import RLAgent

if __name__ == '__main__':
    caro = Caro(size=9, size_to_win=5)
    smart_agent = RLAgent(caro, Cell.X)
    dumb_agent = RLAgent(caro, Cell.O)
    while True:
        print(caro)
        print("=" * (2 * caro.size - 1))
        if caro.turn == smart_agent.cell:
            row, col = smart_agent.generate_optimal_move()
        else:
            row, col = dumb_agent.generate_optimal_move()
            
        caro.move(row, col)
        result = caro.check_win()
        if result != Result.PENDING:
            print(caro)
            match result:
                case Result.X_WIN:
                    print('X wins')
                case Result.O_WIN:
                    print('O wins')
                case Result.DRAW:
                    print('Draw')
            break

