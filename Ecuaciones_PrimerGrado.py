from __future__ import annotations
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Optional, Tuple
from sympy import Symbol, expand, simplify, nsimplify
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)
x = Symbol("x")
TRANSFORMACIONES = standard_transformations + \
    (implicit_multiplication_application,)


def limpiar_simbolos(texto: str) -> str:
    return (texto or "").translate(str.maketrans({
        "−": "-",
        "–": "-",
        "—": "-",
        "×": "*",
        "·": "*",
        "÷": "/",
    }))


def Expresiones(expr, decimales: int = 6) -> str:
    try:
        expr2 = nsimplify(expr)
    except Exception:
        expr2 = expr
    try:
        expr3 = expr2.evalf(decimales)
    except Exception:
        expr3 = expr2
    s = str(expr3)
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def parsear_lado(texto: str):
    texto = limpiar_simbolos(texto)
    texto = texto.replace(" ", "").replace("X", "x").replace("^", "**")
    return parse_expr(texto, transformations=TRANSFORMACIONES, local_dict={"x": x})


def obtener_coeficiente_y_constante(expr) -> Optional[Tuple[object, object]]:
    expr = simplify(expr)
    a = simplify(expr.subs(x, 1) - expr.subs(x, 0))
    b = simplify(expr.subs(x, 0))
    if simplify(expr - (a * x + b)) != 0:
        return None
    return a, b


def normalizar_entrada(ecuacion: str) -> str:
    ecuacion = limpiar_simbolos((ecuacion or "").strip())
    if not ecuacion:
        return ""
    ecuacion = ecuacion.replace("X", "x").replace(" ", "")
    if "=" not in ecuacion:
        ecuacion = f"{ecuacion}=0"
    return ecuacion


@dataclass(frozen=True)
class ResultadoLineal:
    es_lineal: bool
    pasos: str
    solucion: Optional[object]


def resolver_ecuacion_paso_a_paso(ecuacion: str) -> ResultadoLineal:
    ecuacion = normalizar_entrada(ecuacion)
    if not ecuacion:
        raise ValueError("Escribe una ecuación o expresión.")

    texto_izq, texto_der = ecuacion.split("=", 1)
    izq_original = parsear_lado(texto_izq)
    der_original = parsear_lado(texto_der)

    salida = []
    salida.append("1) Ecuación dada:")
    salida.append(
        f"   {Expresiones(izq_original)} = {Expresiones(der_original)}\n")

    izq = expand(izq_original)
    der = expand(der_original)
    salida.append("2) Expandir y simplificar:")
    salida.append(f"   {Expresiones(izq)} = {Expresiones(der)}\n")

    coef_izq = obtener_coeficiente_y_constante(izq)
    coef_der = obtener_coeficiente_y_constante(der)

    if coef_izq is None or coef_der is None:
        salida.append("3) No es una ecuación lineal (de 1er grado).")
        return ResultadoLineal(False, "\n".join(salida), None)

    a_izq, b_izq = coef_izq
    a_der, b_der = coef_der
    A = simplify(a_izq - a_der)
    B = simplify(b_der - b_izq)

    salida.append("3) Pasar términos semejantes:")
    salida.append(f"   {Expresiones(A)}x = {Expresiones(B)}\n")
    if A == 0:
        if B == 0:
            salida.append("Infinitas soluciones (identidad).")
        else:
            salida.append("No tiene solución (contradicción).")
        return ResultadoLineal(True, "\n".join(salida), None)

    solucion = simplify(B / A)
    salida.append("4) Despejar x:")
    salida.append(f"   x = {Expresiones(solucion)}")
    salida.append("\nResultado final:")
    salida.append(f"   x = {Expresiones(solucion)}")
    return ResultadoLineal(True, "\n".join(salida), solucion)


class App(ttk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master, padding=14)
        self.master = master
        self.ecuacion_var = tk.StringVar()
        self.estado_var = tk.StringVar(value="Listo.")
        self.Construir_UI()
        self.Atajos_teclado()

    def Construir_UI(self):
        self.pack(fill="both", expand=True)
        ttk.Label(
            self,
            text="Calculadora de ecuaciones de 1er grado",
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            self,
            text="Si no escribes '=', se asumirá '= 0'",
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(2, 12))
        fila = ttk.Frame(self)
        fila.pack(fill="x")
        ttk.Label(fila, text="Ecuación:").pack(side="left")
        self.entrada = ttk.Entry(fila, textvariable=self.ecuacion_var)
        self.entrada.pack(side="left", fill="x", expand=True, padx=(10, 10))
        self.entrada.focus_set()

        ttk.Button(fila, text="Resolver",
                   command=self.Resolver).pack(side="left")
        ttk.Button(fila, text="Limpiar", command=self.Limpiar).pack(
            side="left", padx=(8, 0))
        ttk.Label(self, textvariable=self.estado_var, font=(
            "Segoe UI", 9)).pack(anchor="w", pady=(8, 8))

        self.salida = ScrolledText(self, wrap="word", height=18)
        self.salida.pack(fill="both", expand=True)
        self.salida.configure(font=("Consolas", 12))

    def Atajos_teclado(self):
        self.master.bind("<Return>", lambda _e: self.Resolver())
        self.master.bind("<Control-l>", lambda _e: self.Limpiar())

    def Resolver(self):
        texto = self.ecuacion_var.get()
        if not texto.strip():
            messagebox.showwarning("Aviso", "Escribe una ecuación.")
            return
        try:
            resultado = resolver_ecuacion_paso_a_paso(texto)
            self.salida.delete("1.0", tk.END)
            self.salida.insert(tk.END, resultado.pasos)
            if resultado.solucion is not None:
                self.estado_var.set(
                    f"Solución: x = {Expresiones(resultado.solucion)}")
            else:
                self.estado_var.set("Proceso completado.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def Limpiar(self):
        self.ecuacion_var.set("")
        self.salida.delete("1.0", tk.END)
        self.estado_var.set("Listo.")
        self.entrada.focus_set()


def iniciar_app():
    root = tk.Tk()
    root.title("Ecuaciones de 1er grado")
    root.geometry("900x580")
    App(root)
    root.mainloop()


if __name__ == "__main__":
    iniciar_app()
