module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<2x2xf32>, %arg1: tensor<f32>, %arg2: tensor<2x2xf32>) -> (tensor<f32>, tensor<i32>) {
    %0 = stablehlo.constant dense<0.0> : tensor<f32>
    %1 = stablehlo.compare LE, %arg1, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %2 = stablehlo.constant dense<0> : tensor<i32>
    %3 = stablehlo.constant dense<9> : tensor<i32>
    %4 = stablehlo.select %1, %3, %2 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %5 = stablehlo.cholesky %arg0, lower = true : tensor<2x2xf32>
    %6 = stablehlo.iota dim = 0 : tensor<2x2xf32>
    %7 = stablehlo.iota dim = 1 : tensor<2x2xf32>
    %8 = stablehlo.compare EQ, %6, %7 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
    %9 = stablehlo.constant dense<0.0> : tensor<2x2xf32>
    %10 = stablehlo.select %8, %5, %9 : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.reduce(%10 init: %11) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %13 = stablehlo.cholesky %arg2, lower = true : tensor<2x2xf32>
    %14 = stablehlo.select %8, %13, %9 : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %15 = stablehlo.reduce(%14 init: %11) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %16 = stablehlo.log %15 : tensor<2xf32>
    %17 = stablehlo.reduce(%16 init: %11) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %18 = stablehlo.constant dense<2.0> : tensor<f32>
    %19 = stablehlo.multiply %18, %17 : tensor<f32>
    %20 = stablehlo.log %12 : tensor<2xf32>
    %21 = stablehlo.reduce(%20 init: %11) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %22 = stablehlo.multiply %18, %21 : tensor<f32>
    %23 = stablehlo.subtract %22, %22 : tensor<f32>
    %24 = stablehlo.compare EQ, %23, %23 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %25 = stablehlo.select %24, %2, %3 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %26 = stablehlo.compare EQ, %4, %2, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %27 = stablehlo.select %26, %25, %4 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %28 = "stablehlo.triangular_solve"(%5, %13) <{left_side = true, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %29 = stablehlo.multiply %28, %28 : tensor<2x2xf32>
    %30 = stablehlo.reduce(%29 init: %11) applies stablehlo.add across dimensions = [0] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %31 = stablehlo.reduce(%30 init: %11) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %32 = stablehlo.constant dense<0.5> : tensor<f32>
    %33 = stablehlo.constant dense<3.0> : tensor<f32>
    %34 = stablehlo.subtract %arg1, %33 : tensor<f32>
    %35 = stablehlo.multiply %32, %34 : tensor<f32>
    %36 = stablehlo.multiply %35, %19 : tensor<f32>
    %37 = stablehlo.constant dense<-0.5> : tensor<f32>
    %38 = stablehlo.multiply %37, %31 : tensor<f32>
    %39 = stablehlo.multiply %arg1, %18 : tensor<f32>
    %40 = stablehlo.constant dense<0.6931471805599453> : tensor<f32>
    %41 = stablehlo.multiply %39, %40 : tensor<f32>
    %42 = stablehlo.multiply %37, %41 : tensor<f32>
    %43 = stablehlo.multiply %32, %arg1 : tensor<f32>
    %44 = stablehlo.negate %43 : tensor<f32>
    %45 = stablehlo.multiply %44, %22 : tensor<f32>
    %46 = stablehlo.constant dense<0.5723649429247001> : tensor<f32>
    %47 = stablehlo.add %43, %0 : tensor<f32>
    %48 = chlo.lgamma %47 : tensor<f32> -> tensor<f32>
    %49 = stablehlo.add %46, %48 : tensor<f32>
    %50 = stablehlo.add %43, %37 : tensor<f32>
    %51 = chlo.lgamma %50 : tensor<f32> -> tensor<f32>
    %52 = stablehlo.add %49, %51 : tensor<f32>
    %53 = stablehlo.negate %52 : tensor<f32>
    %54 = stablehlo.add %36, %38 : tensor<f32>
    %55 = stablehlo.add %54, %42 : tensor<f32>
    %56 = stablehlo.add %55, %45 : tensor<f32>
    %57 = stablehlo.add %56, %53 : tensor<f32>
    return %57, %27 : tensor<f32>, tensor<i32>
  }
}
