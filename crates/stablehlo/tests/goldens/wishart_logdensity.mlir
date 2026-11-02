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
    %14 = stablehlo.iota dim = 0 : tensor<2x2xf32>
    %15 = stablehlo.iota dim = 1 : tensor<2x2xf32>
    %16 = stablehlo.compare EQ, %14, %15 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
    %17 = stablehlo.select %16, %13, %9 : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %18 = stablehlo.reduce(%17 init: %11) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %19 = stablehlo.log %18 : tensor<2xf32>
    %20 = stablehlo.reduce(%19 init: %11) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %21 = stablehlo.constant dense<2.0> : tensor<f32>
    %22 = stablehlo.multiply %21, %20 : tensor<f32>
    %23 = stablehlo.log %12 : tensor<2xf32>
    %24 = stablehlo.reduce(%23 init: %11) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %25 = stablehlo.multiply %21, %24 : tensor<f32>
    %26 = stablehlo.subtract %25, %25 : tensor<f32>
    %27 = stablehlo.compare EQ, %26, %26 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %28 = stablehlo.select %27, %2, %3 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %29 = stablehlo.compare EQ, %4, %2, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %30 = stablehlo.select %29, %28, %4 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %31 = "stablehlo.triangular_solve"(%5, %13) <{left_side = true, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %32 = stablehlo.multiply %31, %31 : tensor<2x2xf32>
    %33 = stablehlo.reduce(%32 init: %11) applies stablehlo.add across dimensions = [0] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %34 = stablehlo.reduce(%33 init: %11) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %35 = stablehlo.constant dense<0.5> : tensor<f32>
    %36 = stablehlo.constant dense<3.0> : tensor<f32>
    %37 = stablehlo.subtract %arg1, %36 : tensor<f32>
    %38 = stablehlo.multiply %35, %37 : tensor<f32>
    %39 = stablehlo.multiply %38, %22 : tensor<f32>
    %40 = stablehlo.constant dense<-0.5> : tensor<f32>
    %41 = stablehlo.multiply %40, %34 : tensor<f32>
    %42 = stablehlo.multiply %arg1, %21 : tensor<f32>
    %43 = stablehlo.constant dense<0.6931471805599453> : tensor<f32>
    %44 = stablehlo.multiply %42, %43 : tensor<f32>
    %45 = stablehlo.multiply %40, %44 : tensor<f32>
    %46 = stablehlo.multiply %35, %arg1 : tensor<f32>
    %47 = stablehlo.negate %46 : tensor<f32>
    %48 = stablehlo.multiply %47, %25 : tensor<f32>
    %49 = stablehlo.constant dense<0.5723649429247001> : tensor<f32>
    %50 = stablehlo.add %46, %0 : tensor<f32>
    %51 = chlo.lgamma %50 : tensor<f32> -> tensor<f32>
    %52 = stablehlo.add %49, %51 : tensor<f32>
    %53 = stablehlo.add %46, %40 : tensor<f32>
    %54 = chlo.lgamma %53 : tensor<f32> -> tensor<f32>
    %55 = stablehlo.add %52, %54 : tensor<f32>
    %56 = stablehlo.negate %55 : tensor<f32>
    %57 = stablehlo.add %39, %41 : tensor<f32>
    %58 = stablehlo.add %57, %45 : tensor<f32>
    %59 = stablehlo.add %58, %48 : tensor<f32>
    %60 = stablehlo.add %59, %56 : tensor<f32>
    return %60, %30 : tensor<f32>, tensor<i32>
  }
}
