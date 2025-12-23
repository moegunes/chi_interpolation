# Interpolating $\chi(r)$
We would like to interpolate the interacting response by using as much knowledge from the non-interacting case as we can. We already know that it satisfies certain behaviors, such as $\chi(q) \propto 1/q^2$ for large-$q$ and $\chi(q) \propto/q^2$ low-$q$. Therefore, the intermediate regime constitutes the key region in which interaction effects become important. These points translates to real space the same way. The large-$r$ behavior is exactly known \cite{Simione2005}, as well as diverging tail as $r\rightarrow 0$ which produces the $1/q^2$ behavior. Thus, we now focus on the intermediate-$r$ region. From Figure \ref{fig:int_vs_nonint} we can already infer that the relevant regime for us would be $k_Fr \simeq [0,6]$. 


The simplest idea we can try is to subtract the asymptotic part from the full expression of $\chi(r)$, i.e. $\chi(r)+\mathcal{B} \frac{\cos{(2k_F r)}}{(2k_F r)^3}$. However, this does not remove the oscillations fully in the large-$r$ region. Simple inspection shows that there is a residual $\mathcal{B} \frac{\sin{(2k_F r)}}{(2k_F r)^4}$ factor as well. After further removing, we are left with a function that does not oscillate or its oscillations are greatly reduced. It is also interesting to note that the removed piece is nothing but the non-interacting response with its prefactor modified by the interacting response function's asymptotic form, i.e.

$$
\chi(r) +\mathcal{B} \frac{\cos{(2k_F r)}}{(2k_F r)^3} - \mathcal{B} \frac{\sin{(2k_F r)}}{(2k_F r)^4} = \chi(r) - \frac{\mathcal{B}}{-6\pi n_0 N_F}\chi_0(r)
$$

However, above equation still is not a good candidate to start interpolating since it diverges as $r \rightarrow 0$. One can then try to scale the above equation with $(2k_Fr)^4$ to get rid of the divergence at $r=0$, however that would yield a second term that is $\mathcal{B} (\sin x -x\cos x)$ which blows up asymptotically. Now, we propose the following form that performs much better,

$$
\mathcal{X}(r)=\frac{(2 k_F r)^4}{r^\gamma}\left(\chi(r)-\frac{\mathcal{B}}{-6\pi n_0 N_F}\,\chi_0(r)\right).
$$

With this definition, we see that there are no more divergences, i.e. $\mathcal{X}(r\rightarrow 0) = 0$ and $\mathcal{X}(r\rightarrow \infty) = 0$. This is due to both scaling with $(2k_Fr)^4$, as well as adding a $1/r^\gamma$ factor which dampens the $\mathcal{B} (\sin x -x\cos x)$ part. For this part, we choose $\gamma=1$ but we will show that other choices can be done and can make optimization more stable. We note that $\mathcal{X}(r)$ decays much faster than $1/r$ which suggests an exponential fitting might be favorable. Lastly, it is straightforward to show that for no interaction, $\mathcal{X}(r)=0$.

## Real-space ansatz for $\mathcal{X}(r)$

Figure \ref{exact_X} shows us that $\mathcal{X}(r)$ is a well behaved, smooth function. To appreciate this function more, we realize that the main peaks around $r=2k_F$ for both $r_s=1$ and $r_s=5$ correspond to the difference between interacting and non-interacting case (see Figure \ref{fig:int_vs_nonint}). From Figure \ref{fig:int_vs_nonint}, we can observe that after this peak, the two response functions quickly converge together. This shows that the main correction comes from the first peak of $\mathcal{X}(r)$, as well as its correct physical limits.

Inspired by above observations, we propose the following ansatz,

$$
    \mathcal{X}(r)=r ^{3-\gamma}\sum_{m=0}^M A_m \cos{(k_mr +\phi_m)}e^{-\alpha_m r}  
$$

with $\gamma=1,2$ and $\alpha_m>0$. We choose this ansatz to have the correct limits (exponential damping for large-$r$ limit and $r^{3-\gamma}$ for $r\rightarrow 0$ limit), as well as the description of the main peak which has an oscillatory behavior. In our optimization, we will focus on finding $\mathcal{X}(r)$ by choosing a small range of $k_Fr$.

## Sum rules for $\mathcal{X}(r)$
For a good description of $\mathcal{X}(r)$, we would need to optimize $4M$ parameters which may seem high. To minimize the number of parameters used, we exploit some exact constraints of both $\chi(r)$ and $\chi^0(r)$. Consider the moments of $\chi(r)$,

$$
\begin{aligned}
C_n
&= \int_0^\infty r^{2n+2}\,\chi(r)\,dr \\
&= \int_0^\infty r^{2n+2}\,
\mathcal{X}(r)\,\frac{r^\gamma}{(2k_F r)^4}\,dr
\;+\;
\frac{\mathcal{B}}{-6\pi n_0 N_F}
\int_0^\infty r^{2n+2}\chi_0(r)dr \\
&= \frac{1}{(2k_F)^4}
\sum_{m=0}^M A_m
\int_0^\infty r^{2n+1}
\cos\!\left(k_m r + \phi_m\right)
e^{-\alpha_m r}dr
+
\frac{\mathcal{B}}{-6\pi n_0 N_F}C_n^{0}.
\end{aligned}
$$

Defining the auxiliary integrals and recalling their closed-form solution,
$$
\begin{aligned}
I_n^m \;\equiv\;
I_n(k_m,\phi_m,\alpha_m)
&=
\int_0^\infty
r^{2n+1}
\cos\!\left(k_m r + \phi_m\right)
e^{-\alpha_m r}\,dr \\
&=
(2n+1)!\;
\Re\!\left[
\frac{e^{i\phi_m}}
{(\alpha_m - i k_m)^{\,2n+2}}
\right].
\end{aligned}
$$ 

We know have the following sum rule for $\mathcal{X}(r)$. Combining Equations \eqref{C_n_derivation} and \eqref{I_n_m},

$$
     \sum_{m=0}^MA_mI^m_n= 16k_F^4\Big(C_n -\frac{\mathcal{B}}{-6\pi n_0N_F} C^0_n\Big) \equiv C^\mathcal{X}_n.
$$

This gives us \(N\) constraints that we can apply to \(4M\) degrees of freedom. However, these constraints are linear only in the amplitudes \(\{A_m\}\); 
the remaining parameters \(\{k_m,\alpha_m,\phi_m\}\) enter the moments through 
the nonlinear dependence of \(I_n^m\). As a consequence, the sum rules can be 
used at this stage to fix the amplitudes uniquely for any prescribed choice of 
the nonlinear parameters, while determining the latter requires solving a 
coupled nonlinear problem.

For a given set of moment indices $(n_j)_{j=1}^N$, it is convenient to introduce
the moment vector $\mathbf{C}^{\mathcal X}$ and the amplitude vector $\mathbf A$,
$$
\mathbf{C}^{\mathcal X}=\begin{pmatrix}C^{\mathcal X}_{n_1} \\ C^{\mathcal X}_{n_2} \\ \vdots \\ C^{\mathcal X}_{n_N} \end{pmatrix}, \qquad \mathbf A = \begin{pmatrix} A_0 \\ A_1 \\ \vdots \\ A_M\end{pmatrix}.
$$

We also define the moment matrix $\mathbf I$ with elements
$$
\bigl[\mathbf I\bigr]_{j m}\equiv I_{n_j}^m\!\left(k_m,\phi_m,\alpha_m\right).
$$

The sum rules then take the compact matrix form
$$
\mathbf I\,\mathbf A=\mathbf C^{\mathcal X}.
$$
For fixed \(\{k_m,\alpha_m,\phi_m\}\), Eq.~\eqref{eq:matrix_sum_rules} constitutes 
a linear system for the amplitudes \(\{A_m\}\), which can be solved exactly when \(N=M+1\), or in a least-squares sense when \(N>M+1\). In the remainder of this 
section, we therefore focus on the determination of the amplitudes from the sum 
rules, and defer the discussion of how the nonlinear parameters are fixed to a 
later section.

Increasing the number of enforced moments \(N\) would in principle improve the representation. However, each additional mode introduces three new nonlinear parameters \((k_m,\alpha_m,\phi_m)\) in addition to the linear amplitude \(A_m\), so that increasing \(N\) must be accompanied by an increase in the number of modes \(M\). This rapidly enlarges the nonlinear parameter space and limits the practical number of constraints that can be imposed. We therefore adopt the minimal nontrivial choice \(M=1\) (two modes) and enforce \(N=2\) moment constraints, which fixes the amplitudes uniquely while keeping the number of nonlinear parameters manageable.
