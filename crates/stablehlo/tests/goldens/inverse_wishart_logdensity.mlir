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
    %14 = stablehlo.log %12 : tensor<2xf32>
    %15 = stablehlo.reduce(%14 init: %11) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %16 = stablehlo.constant dense<2.0> : tensor<f32>
    %17 = stablehlo.multiply %16, %15 : tensor<f32>
    %18 = stablehlo.subtract %17, %17 : tensor<f32>
    %19 = stablehlo.compare EQ, %18, %18 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %20 = stablehlo.select %19, %2, %3 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %21 = stablehlo.compare EQ, %4, %2, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %22 = stablehlo.select %21, %20, %4 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %23 = stablehlo.select %8, %13, %9 : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %24 = stablehlo.reduce(%23 init: %11) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %25 = stablehlo.log %24 : tensor<2xf32>
    %26 = stablehlo.reduce(%25 init: %11) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %27 = stablehlo.multiply %16, %26 : tensor<f32>
    %28 = "stablehlo.triangular_solve"(%13, %5) <{left_side = true, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %29 = stablehlo.multiply %28, %28 : tensor<2x2xf32>
    %30 = stablehlo.reduce(%29 init: %11) applies stablehlo.add across dimensions = [0] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %31 = stablehlo.reduce(%30 init: %11) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %32 = stablehlo.constant dense<0.5> : tensor<f32>
    %33 = stablehlo.multiply %32, %arg1 : tensor<f32>
    %34 = stablehlo.multiply %33, %17 : tensor<f32>
    %35 = stablehlo.constant dense<3.0> : tensor<f32>
    %36 = stablehlo.add %arg1, %35 : tensor<f32>
    %37 = stablehlo.constant dense<-0.5> : tensor<f32>
    %38 = stablehlo.multiply %37, %36 : tensor<f32>
    %39 = stablehlo.multiply %38, %27 : tensor<f32>
    %40 = stablehlo.multiply %37, %31 : tensor<f32>
    %41 = stablehlo.multiply %arg1, %16 : tensor<f32>
    %42 = stablehlo.constant dense<0.6931471805599453> : tensor<f32>
    %43 = stablehlo.multiply %41, %42 : tensor<f32>
    %44 = stablehlo.multiply %37, %43 : tensor<f32>
    %45 = stablehlo.constant dense<0.5723649429247001> : tensor<f32>
    %46 = stablehlo.add %33, %0 : tensor<f32>
    %47 = chlo.lgamma %46 : tensor<f32> -> tensor<f32>
    %48 = stablehlo.add %45, %47 : tensor<f32>
    %49 = stablehlo.add %33, %37 : tensor<f32>
    %50 = chlo.lgamma %49 : tensor<f32> -> tensor<f32>
    %51 = stablehlo.add %48, %50 : tensor<f32>
    %52 = stablehlo.negate %51 : tensor<f32>
    %53 = stablehlo.add %34, %39 : tensor<f32>
    %54 = stablehlo.add %53, %40 : tensor<f32>
    %55 = stablehlo.add %54, %44 : tensor<f32>
    %56 = stablehlo.add %55, %52 : tensor<f32>
    return %56, %22 : tensor<f32>, tensor<i32>
  }
}
