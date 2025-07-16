import tkinter as tk
from tkinter import messagebox
from game import GomokuGame
from ai_pytorch_resnet_v2 import GomokuAI_PyTorch  # <-- 导入 PyTorch AI


class GomokuGUI(tk.Tk):
    """
    五子棋图形用户界面 (GUI)
    """

    def __init__(self, game, ai):
        super().__init__()
        self.game = game
        self.ai = ai
        self.title("9x9 五子棋 (PyTorch 版)")
        self.resizable(False, False)

        # 设置棋盘尺寸
        self.board_size = game.size
        self.cell_size = 50
        self.margin = 30
        canvas_size = self.board_size * self.cell_size + 2 * self.margin

        # 创建画布
        self.canvas = tk.Canvas(self, width=canvas_size, height=canvas_size, bg="#F0C080")
        self.canvas.pack(pady=10, padx=10)

        # 创建按钮
        self.restart_button = tk.Button(self, text="重新开始", command=self.restart_game, font=("Arial", 14))
        self.restart_button.pack(pady=10)

        # 绑定鼠标点击事件
        self.canvas.bind("<Button-1>", self.handle_click)

        # 绘制棋盘
        self.draw_board()

    def draw_board(self):
        """绘制棋盘网格"""
        start = self.margin + self.cell_size / 2
        end = self.margin + (self.board_size - 0.5) * self.cell_size
        for i in range(self.board_size):
            pos = start + i * self.cell_size
            self.canvas.create_line(start, pos, end, pos)
            self.canvas.create_line(pos, start, pos, end)

    def draw_stones(self):
        """根据游戏状态绘制所有棋子"""
        self.canvas.delete("stone")
        radius = self.cell_size / 2 * 0.85
        for r in range(self.board_size):
            for c in range(self.board_size):
                player = self.game.board[r, c]
                if player != 0:
                    cx = self.margin + (c + 0.5) * self.cell_size
                    cy = self.margin + (r + 0.5) * self.cell_size
                    color = "black" if player == 1 else "white"
                    self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, fill=color,
                                            outline=color, tags="stone")

    def handle_click(self, event):
        """处理玩家的鼠标点击"""
        if self.game.game_over or self.game.current_player != 1:
            return

        col = round((event.x - self.margin - self.cell_size / 2) / self.cell_size)
        row = round((event.y - self.margin - self.cell_size / 2) / self.cell_size)

        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            if self.game.make_move(row, col):
                self.draw_stones()
                self.update()
                if not self.game.game_over:
                    self.after(500, self.ai_turn)
                else:
                    self.show_winner()


    def ai_turn(self):
        """处理AI的下棋回合"""
        if self.game.game_over or self.game.current_player != 2:
            return

        # 注意：这里传递的是整个game实例，因为AI需要知道当前玩家
        move = self.ai.find_best_move(self.game)

        if move:
            self.game.make_move(move[0], move[1])
            self.draw_stones()
            if self.game.game_over:
                self.show_winner()

    def restart_game(self):
        """重新开始一局新游戏"""
        self.game.reset()
        self.draw_stones()
        messagebox.showinfo("游戏重置", "新游戏开始！您执黑棋先手。")

    def show_winner(self):
        """游戏结束后显示获胜信息"""
        if self.game.winner == 1:
            message = "恭喜你，你赢了！"
        elif self.game.winner == 2:
            message = "很遗憾，AI获胜！"
        else:
            message = "平局！"
        messagebox.showinfo("游戏结束", message)


if __name__ == "__main__":
    game_instance = GomokuGame(size=9)
    # 使用 PyTorch AI 实例
    ai_instance = GomokuAI_PyTorch(board_size=9)
    app = GomokuGUI(game_instance, ai_instance)
    app.mainloop()